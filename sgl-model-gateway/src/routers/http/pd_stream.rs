//! Opt-in HTTP PD streaming. A real decode result confirms KV handoff; a
//! subsequent prefill HTTP failure must not invalidate that result.
//!
//! Until handoff (or prefill completion), both upstreams stay in the handler.
//! Afterwards one bounded relay task also drains prefill. Its admission permit
//! remains held through that drain, including when decode has already finished.

use std::{future::pending, sync::Arc};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::{future::BoxFuture, stream, StreamExt};
use serde_json::Value;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::{
    core::{AttachedBody, Worker, WorkerLoadGuard},
    observability::{
        events::{self, Event},
        metrics::{metrics_labels, Metrics},
    },
    routers::{
        grpc::utils::error_type_from_status, header_utils, streaming_utils::BreakerTrackedStream,
    },
};

const ENABLE_ENV: &str = "SMG_PD_EARLY_DECODE_STREAM";
const MAX_PROBE_BYTES: usize = 256 * 1024;
const RELAY_CAPACITY: usize = 4;

#[derive(Debug, Default)]
pub(super) struct EarlyStream {
    slots: Option<Arc<Semaphore>>,
}

impl EarlyStream {
    pub(super) fn from_env(max_concurrent: i32) -> Result<Self, String> {
        let enabled = match std::env::var(ENABLE_ENV) {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "0" => false,
            Ok(value) if value == "1" => true,
            _ => return Err(format!("{ENABLE_ENV} must be 0 or 1")),
        };
        if !enabled {
            return Ok(Self::default());
        }
        // Include post-decode drains in the cap. A full cap falls back to the
        // existing dispatch before either request is sent; it never rejects or
        // duplicates a request. Unlimited router admission still needs a cap.
        let limit = if max_concurrent > 0 {
            max_concurrent as usize
        } else {
            1024
        };
        tracing::info!(limit, "HTTP PD early decode stream enabled (handoff-v1)");
        Ok(Self {
            slots: Some(Arc::new(Semaphore::new(limit))),
        })
    }

    pub(super) fn try_acquire(
        &self,
        route: &str,
        is_stream: bool,
        return_logprob: bool,
        batch_size: Option<usize>,
        request: &Value,
    ) -> Option<OwnedSemaphorePermit> {
        let slots = self.slots.as_ref()?;
        if !eligible(route, is_stream, return_logprob, batch_size, request) {
            return None;
        }
        Arc::clone(slots).try_acquire_owned().ok()
    }
}

fn eligible(
    route: &str,
    is_stream: bool,
    return_logprob: bool,
    batch_size: Option<usize>,
    request: &Value,
) -> bool {
    matches!(
        route,
        "/generate" | "/v1/completions" | "/v1/chat/completions"
    ) && is_stream
        && !return_logprob
        && batch_size.is_none()
        && request.get("n").and_then(Value::as_u64).unwrap_or(1) == 1
        && request.get("best_of").and_then(Value::as_u64).unwrap_or(1) == 1
        && request
            .pointer("/sampling_params/n")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            == 1
        && request
            .get("tools")
            .is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty))
}

pub(super) enum DispatchError {
    Prefill(Result<reqwest::Response, reqwest::Error>),
    Decode(Result<reqwest::Response, reqwest::Error>),
}

type PrefillResult = Result<(), Result<reqwest::Response, reqwest::Error>>;
type PrefillFuture = BoxFuture<'static, PrefillResult>;

fn record_error(side: &'static str, status: StatusCode) {
    if status.is_server_error() {
        Metrics::record_worker_error(
            side,
            metrics_labels::CONNECTION_HTTP,
            error_type_from_status(status),
        );
    }
}

fn watch_prefill(
    request: reqwest::RequestBuilder,
    worker: Arc<dyn Worker>,
    guard: WorkerLoadGuard,
) -> PrefillFuture {
    Box::pin(async move {
        let _guard = guard;
        let response = match request.send().await {
            Ok(response) => response,
            Err(err) => {
                worker.record_outcome(false);
                record_error(metrics_labels::WORKER_PREFILL, StatusCode::BAD_GATEWAY);
                return Err(Err(err));
            }
        };
        let status = response.status();
        if !status.is_success() {
            worker.record_outcome(status.is_client_error());
            record_error(metrics_labels::WORKER_PREFILL, status);
            return Err(Ok(response));
        }
        // No logprobs are needed here. Drain without retaining the whole body.
        // Preserve process_prefill_response(false)'s warn-only body-read error
        // semantics after a successful status. reqwest keeps the original total
        // request deadline while this stream is being read.
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            if let Err(err) = chunk {
                warn!(prefill_url = worker.url(), %err, "Prefill response body read failed");
                break;
            }
        }
        worker.record_outcome(true);
        Ok(())
    })
}

async fn poll_prefill(watch: &mut Option<PrefillFuture>) -> PrefillResult {
    match watch {
        Some(watch) => watch.await,
        None => pending().await,
    }
}

fn report_late_prefill(result: PrefillResult, url: &str) {
    match result {
        Ok(()) => {}
        Err(Ok(response)) => warn!(
            prefill_url = url, status = %response.status(),
            "Prefill HTTP failed after decode handoff; decode remains authoritative"
        ),
        Err(Err(err)) => warn!(
            prefill_url = url, %err,
            "Prefill HTTP failed after decode handoff; decode remains authoritative"
        ),
    }
}

/// No task is spawned until we have prefill success or a real decode result.
/// Dropping this future drops BOTH upstream requests, including during retries.
pub(super) async fn dispatch(
    prefill_request: reqwest::RequestBuilder,
    decode_request: reqwest::RequestBuilder,
    route: &'static str,
    headers: &HeaderMap,
    prefill: Arc<dyn Worker>,
    decode: Arc<dyn Worker>,
    slot: OwnedSemaphorePermit,
) -> Result<Response, DispatchError> {
    let prefill_guard = WorkerLoadGuard::new(Arc::clone(&prefill), Some(headers));
    let decode_guard = WorkerLoadGuard::new(Arc::clone(&decode), Some(headers));
    let mut watch = Some(watch_prefill(
        prefill_request,
        Arc::clone(&prefill),
        prefill_guard,
    ));
    let decode_fut = decode_request.send();
    tokio::pin!(decode_fut);
    let response = loop {
        tokio::select! {
            biased;
            result = &mut decode_fut => break result.map_err(|err| DispatchError::Decode(Err(err)))?,
            result = poll_prefill(&mut watch) => {
                watch = None;
                result.map_err(DispatchError::Prefill)?;
            }
        }
    };
    events::RequestReceivedEvent {}.emit();
    if !response.status().is_success() {
        return Err(DispatchError::Decode(Ok(response)));
    }
    let status = response.status();
    let response_headers = header_utils::preserve_response_headers(response.headers());
    let mut upstream = response.bytes_stream();
    let mut buffered = Vec::new();
    let mut probe = HandoffProbe::new(route);
    let mut can_probe = true;
    let mut event_error = None;

    while watch.is_some() {
        tokio::select! {
            biased;
            chunk = upstream.next(), if can_probe => {
                match chunk {
                    Some(Ok(chunk)) => {
                        if chunk.is_empty() {
                            continue;
                        }
                        let decision = probe.feed(&chunk);
                        buffered.push(chunk);
                        match decision {
                            ProbeResult::Handoff => break,
                            ProbeResult::Error(worker_fault) => {
                                // A D error event is not handoff success. Cancel P
                                // and forward the original error, without rewriting it.
                                watch = None;
                                event_error = Some(worker_fault);
                                break;
                            }
                            ProbeResult::WaitPrefill => can_probe = false,
                            ProbeResult::Pending => {}
                        }
                        if buffered.len() >= 64 {
                            can_probe = false;
                        }
                    }
                    Some(Err(err)) => return Err(DispatchError::Decode(Err(err))),
                    // No proof of handoff: preserve the old P gate for empty or
                    // unrecognised streams. In particular, 200/role/[DONE] alone
                    // is not a successful KV handoff signal.
                    None => can_probe = false,
                }
            }
            result = poll_prefill(&mut watch) => {
                watch = None;
                result.map_err(DispatchError::Prefill)?;
            }
        }
    }

    let stream = stream::iter(buffered.into_iter().map(Ok)).chain(upstream);
    let mut tracked =
        BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_owned());
    if event_error == Some(true) {
        tracked.mark_errored();
        record_error(metrics_labels::WORKER_DECODE, StatusCode::BAD_GATEWAY);
    }
    let (tx, rx) = mpsc::channel(RELAY_CAPACITY);
    let prefill_url = prefill.url().to_owned();
    tokio::spawn(relay(tracked, watch, tx, prefill_url, slot));

    let mut result = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    *result.status_mut() = status;
    *result.headers_mut() = response_headers;
    result
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    Ok(AttachedBody::wrap_response(result, decode_guard))
}

async fn relay(
    mut upstream: BreakerTrackedStream,
    mut watch: Option<PrefillFuture>,
    tx: mpsc::Sender<Result<Bytes, String>>,
    prefill_url: String,
    slot: OwnedSemaphorePermit,
) {
    let _slot = slot;
    let mut pending_chunk = None;
    let mut finished = false;
    let mut failed = false;
    // Detect the protocol terminator across HTTP chunk boundaries, without
    // parsing JSON or allocating on the token path.
    let mut done = DoneScanner::default();
    loop {
        if finished && pending_chunk.is_none() {
            break;
        }
        tokio::select! {
            biased;
            _ = tx.closed() => return,
            result = poll_prefill(&mut watch) => {
                watch = None;
                report_late_prefill(result, &prefill_url);
            }
            permit = tx.reserve(), if pending_chunk.is_some() => {
                match permit {
                    Ok(permit) => {
                        if let Some(chunk) = pending_chunk.take() {
                            permit.send(chunk);
                        }
                    }
                    Err(_) => return,
                }
            }
            chunk = upstream.next(), if !finished && pending_chunk.is_none() => {
                match chunk {
                    Some(Ok(chunk)) => {
                        finished = done.feed(&chunk);
                        if finished {
                            upstream.mark_completed();
                        }
                        pending_chunk = Some(Ok(chunk));
                    }
                    Some(Err(err)) => {
                        record_error(metrics_labels::WORKER_DECODE, StatusCode::BAD_GATEWAY);
                        failed = true;
                        finished = true;
                        pending_chunk = Some(Err(format!("Stream error: {err}")));
                    }
                    None => finished = true,
                }
            }
        }
    }
    // Close the client response BEFORE waiting for P. The same task keeps the
    // admission permit until drain/timeout, so completed D requests cannot
    // accumulate an unlimited number of detached prefill tasks.
    drop(tx);
    drop(upstream);
    if !failed {
        if let Some(watch) = watch {
            report_late_prefill(watch.await, &prefill_url);
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ProbeResult {
    Pending,
    Handoff,
    Error(bool),
    WaitPrefill,
}

struct HandoffProbe {
    route: &'static str,
    line: Vec<u8>,
    data: Vec<u8>,
    inspected: usize,
}

impl HandoffProbe {
    fn new(route: &'static str) -> Self {
        Self {
            route,
            line: Vec::new(),
            data: Vec::new(),
            inspected: 0,
        }
    }

    fn feed(&mut self, mut bytes: &[u8]) -> ProbeResult {
        self.inspected = self.inspected.saturating_add(bytes.len());
        if self.inspected > MAX_PROBE_BYTES {
            return ProbeResult::WaitPrefill;
        }
        while !bytes.is_empty() {
            let end = memchr::memchr(b'\n', bytes);
            let len = end.unwrap_or(bytes.len());
            self.line.extend_from_slice(&bytes[..len]);
            bytes = &bytes[len + usize::from(end.is_some())..];
            if end.is_none() {
                break;
            }
            if self.line.last() == Some(&b'\r') {
                self.line.pop();
            }
            if self.line.is_empty() {
                if !self.data.is_empty() {
                    let result = classify_event(self.route, &self.data);
                    self.data.clear();
                    if result != ProbeResult::Pending {
                        return result;
                    }
                }
            } else if let Some(value) = self.line.strip_prefix(b"data:") {
                let value = value.strip_prefix(b" ").unwrap_or(value);
                if !self.data.is_empty() {
                    self.data.push(b'\n');
                }
                self.data.extend_from_slice(value);
            }
            self.line.clear();
        }
        ProbeResult::Pending
    }
}

fn classify_event(route: &str, data: &[u8]) -> ProbeResult {
    if data == b"[DONE]" {
        return ProbeResult::WaitPrefill;
    }
    let Ok(value) = serde_json::from_slice::<Value>(data) else {
        return ProbeResult::WaitPrefill;
    };
    if let Some(error) = value.get("error").filter(|v| !v.is_null()) {
        let code = error
            .get("code")
            .or_else(|| error.get("status"))
            .and_then(Value::as_u64);
        return ProbeResult::Error(!code.is_some_and(|code| (400..500).contains(&code)));
    }
    if route == "/generate" {
        if value
            .pointer("/meta_info/finish_reason/type")
            .and_then(Value::as_str)
            == Some("abort")
        {
            let code = value
                .pointer("/meta_info/finish_reason/status_code")
                .and_then(Value::as_u64);
            return ProbeResult::Error(!code.is_some_and(|code| (400..500).contains(&code)));
        }
        if value
            .pointer("/meta_info/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
            || value
                .get("output_ids")
                .and_then(Value::as_array)
                .is_some_and(|ids| !ids.is_empty())
        {
            return ProbeResult::Handoff;
        }
    } else if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        if choices.len() == 1 {
            let choice = &choices[0];
            let text = if route == "/v1/completions" {
                choice.get("text")
            } else {
                choice.pointer("/delta/content")
            };
            let reasoning = choice.pointer("/delta/reasoning_content");
            let finish = choice.get("finish_reason").and_then(Value::as_str);
            if text.and_then(Value::as_str).is_some_and(|s| !s.is_empty())
                || reasoning
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty())
                || matches!(finish, Some("stop" | "length" | "tool_calls"))
            {
                return ProbeResult::Handoff;
            }
        }
    }
    ProbeResult::Pending
}

#[derive(Default)]
struct DoneScanner {
    line: [u8; 32],
    len: usize,
    overlong: bool,
    done_data: bool,
    has_data: bool,
}

impl DoneScanner {
    fn feed(&mut self, mut bytes: &[u8]) -> bool {
        while !bytes.is_empty() {
            let end = memchr::memchr(b'\n', bytes);
            let len = end.unwrap_or(bytes.len());
            let take = len.min(self.line.len() - self.len);
            self.line[self.len..self.len + take].copy_from_slice(&bytes[..take]);
            self.len += take;
            self.overlong |= take < len;
            bytes = &bytes[len + usize::from(end.is_some())..];
            if end.is_none() {
                break;
            }
            let line = &self.line[..self.len];
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if !self.overlong && line.is_empty() {
                if self.done_data {
                    return true;
                }
                self.done_data = false;
                self.has_data = false;
            } else if line.starts_with(b"data:") {
                self.done_data = !self.has_data
                    && !self.overlong
                    && std::str::from_utf8(&line[5..]).is_ok_and(|s| s.trim() == "[DONE]");
                self.has_data = true;
            }
            self.len = 0;
            self.overlong = false;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{io, time::Duration};

    use axum::{routing::post, Router};
    use futures_util::FutureExt;
    use tokio::{net::TcpListener, sync::Notify, task::JoinHandle};
    use tokio_stream::wrappers::UnboundedReceiverStream;

    use super::*;
    use crate::core::BasicWorkerBuilder;

    const TOKEN: &str =
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n";
    const ROLE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n";
    const DONE: &str = "data: [DONE]\n\n";

    #[test]
    fn handoff_requires_generation_not_headers_role_or_error() {
        assert_eq!(
            classify_event(
                "/v1/chat/completions",
                b"{\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}"
            ),
            ProbeResult::Pending
        );
        assert_eq!(
            classify_event("/v1/chat/completions", b"{\"error\":{\"code\":400}}"),
            ProbeResult::Error(false)
        );
        assert_eq!(classify_event("/generate", b"{\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":{\"type\":\"abort\",\"status_code\":500}}}"), ProbeResult::Error(true));
        assert_eq!(
            classify_event("/generate", b"{\"output_ids\":[123]}"),
            ProbeResult::Handoff
        );
        assert_eq!(
            classify_event("/generate", b"{\"meta_info\":{\"completion_tokens\":1}}"),
            ProbeResult::Handoff
        );
        assert_eq!(
            classify_event(
                "/v1/completions",
                b"{\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}]}"
            ),
            ProbeResult::Handoff
        );
        assert_eq!(
            classify_event("/v1/chat/completions", b"[DONE]"),
            ProbeResult::WaitPrefill
        );
    }

    #[test]
    fn handoff_probe_handles_every_split_and_crlf() {
        let event = format!(
            ": comment\r\n{}{}",
            ROLE.replace('\n', "\r\n"),
            TOKEN.replace('\n', "\r\n")
        );
        for split in 0..=event.len() {
            let mut probe = HandoffProbe::new("/v1/chat/completions");
            let first = probe.feed(&event.as_bytes()[..split]);
            let result = if first == ProbeResult::Pending {
                probe.feed(&event.as_bytes()[split..])
            } else {
                first
            };
            assert_eq!(result, ProbeResult::Handoff, "split={split}");
        }
        let mut probe = HandoffProbe::new("/generate");
        assert_eq!(
            probe.feed(&vec![b'x'; MAX_PROBE_BYTES + 1]),
            ProbeResult::WaitPrefill
        );
    }

    #[test]
    fn done_scanner_handles_boundaries_and_does_not_match_json_text() {
        let event = format!("{TOKEN}data: [DONE]\r\n\r\n");
        for split in 0..=event.len() {
            let mut scanner = DoneScanner::default();
            assert!(
                scanner.feed(&event.as_bytes()[..split])
                    || scanner.feed(&event.as_bytes()[split..])
            );
        }
        let mut scanner = DoneScanner::default();
        assert!(!scanner.feed(b"data: {\"text\":\"data: [DONE]\"}\n\n"));
        assert!(!scanner.feed(b"data: not done\ndata: [DONE]\n\n"));
        assert!(scanner.feed(b"data:[DONE]\n\n"));
    }

    #[test]
    fn eligibility_and_admission_fall_back_without_waiting() {
        let request = serde_json::json!({"n":1});
        assert!(eligible(
            "/v1/chat/completions",
            true,
            false,
            None,
            &request
        ));
        for (streaming, logprob, batch) in [
            (false, false, None),
            (true, true, None),
            (true, false, Some(1)),
        ] {
            assert!(!eligible("/generate", streaming, logprob, batch, &request));
        }
        assert!(!eligible(
            "/v1/chat/completions",
            true,
            false,
            None,
            &serde_json::json!({"n":2})
        ));
        assert!(!eligible(
            "/generate",
            true,
            false,
            None,
            &serde_json::json!({"sampling_params":{"n":2}})
        ));
        let config = EarlyStream {
            slots: Some(Arc::new(Semaphore::new(1))),
        };
        let slot = config
            .try_acquire("/generate", true, false, None, &request)
            .unwrap();
        assert!(config
            .try_acquire("/generate", true, false, None, &request)
            .is_none());
        drop(slot);
        assert!(config
            .try_acquire("/generate", true, false, None, &request)
            .is_some());
        assert!(EarlyStream::default()
            .try_acquire("/generate", true, false, None, &request)
            .is_none());
    }

    struct Server {
        url: String,
        task: JoinHandle<()>,
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn serve(app: Router) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/generate", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Server { url, task }
    }

    async fn prefill_server(
        status: StatusCode,
        headers: Option<Arc<Notify>>,
        body: Option<Arc<Notify>>,
    ) -> Server {
        serve(Router::new().route(
            "/generate",
            post(move || {
                let headers = headers.clone();
                let body = body.clone();
                async move {
                    if let Some(gate) = headers {
                        gate.notified().await;
                    }
                    let body = Body::from_stream(stream::once(async move {
                        if let Some(gate) = body {
                            gate.notified().await;
                        }
                        Ok::<_, io::Error>(Bytes::from_static(b"{}"))
                    }));
                    let mut response = Response::new(body);
                    *response.status_mut() = status;
                    response
                }
            }),
        ))
        .await
    }

    async fn decode_server() -> (Server, mpsc::UnboundedSender<Result<Bytes, io::Error>>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let state = Arc::new(std::sync::Mutex::new(Some(rx)));
        let server = serve(Router::new().route(
            "/generate",
            post(move || {
                let state = Arc::clone(&state);
                async move {
                    let rx = state.lock().unwrap().take().unwrap();
                    Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)))
                }
            }),
        ))
        .await;
        (server, tx)
    }

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new(url).build())
    }

    async fn eventually(mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while !condition() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    async fn start(
        p: &Server,
        d: &Server,
        pw: Arc<dyn Worker>,
        dw: Arc<dyn Worker>,
        slots: Arc<Semaphore>,
    ) -> Result<Response, DispatchError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        dispatch(
            client.post(&p.url),
            client.post(&d.url),
            "/v1/chat/completions",
            &HeaderMap::new(),
            pw,
            dw,
            slots.acquire_owned().await.unwrap(),
        )
        .await
    }

    #[tokio::test]
    async fn pd_early_stream_token_and_done_do_not_wait_for_prefill_headers_or_body() {
        for delay_headers in [true, false] {
            let gate = Arc::new(Notify::new());
            let p = prefill_server(
                StatusCode::OK,
                delay_headers.then(|| gate.clone()),
                (!delay_headers).then(|| gate.clone()),
            )
            .await;
            let (d, tx) = decode_server().await;
            let pw = worker(&p.url);
            let dw = worker(&d.url);
            let slots = Arc::new(Semaphore::new(1));
            // Fragment the valid event and terminator to test exact replay.
            tx.send(Ok(Bytes::from_static(ROLE.as_bytes()))).unwrap();
            tx.send(Ok(Bytes::from_static(&TOKEN.as_bytes()[..11])))
                .unwrap();
            tx.send(Ok(Bytes::from_static(&TOKEN.as_bytes()[11..])))
                .unwrap();
            tx.send(Ok(Bytes::from_static(b"data: [DO"))).unwrap();
            tx.send(Ok(Bytes::from_static(b"NE]\n\n"))).unwrap();
            let response = tokio::time::timeout(
                Duration::from_secs(2),
                start(&p, &d, pw.clone(), dw.clone(), slots.clone()),
            )
            .await
            .unwrap()
            .ok()
            .unwrap();
            let output = tokio::time::timeout(
                Duration::from_secs(2),
                axum::body::to_bytes(response.into_body(), 8192),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(output.as_ref(), format!("{ROLE}{TOKEN}{DONE}").as_bytes());
            assert_eq!(
                slots.available_permits(),
                0,
                "P drain must retain admission"
            );
            assert_eq!(pw.load(), 1);
            assert_eq!(dw.load(), 0);
            eventually(|| dw.circuit_breaker().total_successes() == 1).await;
            assert_eq!(dw.circuit_breaker().total_successes(), 1);
            gate.notify_one();
            eventually(|| slots.available_permits() == 1).await;
            assert_eq!(pw.load(), 0);
            assert_eq!(pw.circuit_breaker().total_successes(), 1);
        }
    }

    #[tokio::test]
    async fn pd_early_stream_late_prefill_failure_does_not_cancel_decode() {
        let gate = Arc::new(Notify::new());
        let p = prefill_server(StatusCode::INTERNAL_SERVER_ERROR, Some(gate.clone()), None).await;
        let (d, tx) = decode_server().await;
        let pw = worker(&p.url);
        let dw = worker(&d.url);
        let slots = Arc::new(Semaphore::new(1));
        tx.send(Ok(Bytes::from_static(TOKEN.as_bytes()))).unwrap();
        let response = start(&p, &d, pw.clone(), dw.clone(), slots.clone())
            .await
            .ok()
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        assert_eq!(
            body.next().await.unwrap().unwrap().as_ref(),
            TOKEN.as_bytes()
        );
        gate.notify_one();
        eventually(|| pw.circuit_breaker().total_failures() == 1).await;
        tx.send(Ok(Bytes::from_static(TOKEN.as_bytes()))).unwrap();
        tx.send(Ok(Bytes::from_static(DONE.as_bytes()))).unwrap();
        assert_eq!(
            body.next().await.unwrap().unwrap().as_ref(),
            TOKEN.as_bytes()
        );
        assert_eq!(
            body.next().await.unwrap().unwrap().as_ref(),
            DONE.as_bytes()
        );
        assert!(body.next().await.is_none());
        drop(body);
        eventually(|| slots.available_permits() == 1).await;
        assert_eq!(dw.circuit_breaker().total_failures(), 0);
        assert_eq!(dw.circuit_breaker().total_successes(), 1);
    }

    #[tokio::test]
    async fn pd_early_stream_prefill_failure_before_handoff_is_request_local() {
        let gate = Arc::new(Notify::new());
        let p = prefill_server(StatusCode::SERVICE_UNAVAILABLE, Some(gate.clone()), None).await;
        let (d, tx) = decode_server().await;
        let pw = worker(&p.url);
        let dw = worker(&d.url);
        let slots = Arc::new(Semaphore::new(1));
        tx.send(Ok(Bytes::from_static(ROLE.as_bytes()))).unwrap();
        let future = start(&p, &d, pw.clone(), dw.clone(), slots.clone());
        tokio::pin!(future);
        assert!(future.as_mut().now_or_never().is_none());
        gate.notify_one();
        let result = tokio::time::timeout(Duration::from_secs(2), future)
            .await
            .unwrap();
        assert!(matches!(result, Err(DispatchError::Prefill(Ok(_)))));
        assert_eq!(pw.load(), 0);
        assert_eq!(dw.load(), 0);
        assert_eq!(dw.circuit_breaker().total_failures(), 0);
        assert_eq!(slots.available_permits(), 1);
        // Same runtime still accepts unrelated requests after this request fails.
        let healthy = prefill_server(StatusCode::OK, None, None).await;
        assert!(reqwest::Client::new()
            .post(&healthy.url)
            .send()
            .await
            .unwrap()
            .status()
            .is_success());
    }

    #[tokio::test]
    async fn pd_early_stream_client_drop_cancels_both_and_releases_slot() {
        let gate = Arc::new(Notify::new());
        let p = prefill_server(StatusCode::OK, Some(gate), None).await;
        let (d, tx) = decode_server().await;
        let pw = worker(&p.url);
        let dw = worker(&d.url);
        let slots = Arc::new(Semaphore::new(1));
        tx.send(Ok(Bytes::from_static(TOKEN.as_bytes()))).unwrap();
        let response = start(&p, &d, pw.clone(), dw.clone(), slots.clone())
            .await
            .ok()
            .unwrap();
        // Fill the bounded relay while the client isn't reading.
        for _ in 0..64 {
            let _ = tx.send(Ok(Bytes::from_static(TOKEN.as_bytes())));
        }
        drop(response);
        eventually(|| slots.available_permits() == 1).await;
        assert_eq!(pw.load(), 0);
        assert_eq!(dw.load(), 0);
        assert_eq!(pw.circuit_breaker().total_failures(), 0);
        assert_eq!(dw.circuit_breaker().total_failures(), 0);
    }

    #[tokio::test]
    async fn pd_early_stream_handler_drop_does_not_spawn_or_leak_work() {
        let p = prefill_server(StatusCode::OK, Some(Arc::new(Notify::new())), None).await;
        let (d, _tx) = decode_server().await;
        let pw = worker(&p.url);
        let dw = worker(&d.url);
        let slots = Arc::new(Semaphore::new(1));
        {
            let future = start(&p, &d, pw.clone(), dw.clone(), slots.clone());
            tokio::pin!(future);
            assert!(future.as_mut().now_or_never().is_none());
        }
        assert_eq!(slots.available_permits(), 1);
        assert_eq!(pw.load(), 0);
        assert_eq!(dw.load(), 0);
        assert_eq!(pw.circuit_breaker().total_failures(), 0);
        assert_eq!(dw.circuit_breaker().total_failures(), 0);
    }

    #[tokio::test]
    async fn pd_early_stream_post_decode_drain_obeys_original_timeout() {
        let p = prefill_server(StatusCode::OK, Some(Arc::new(Notify::new())), None).await;
        let (d, tx) = decode_server().await;
        let pw = worker(&p.url);
        let dw = worker(&d.url);
        let slots = Arc::new(Semaphore::new(1));
        tx.send(Ok(Bytes::from_static(TOKEN.as_bytes()))).unwrap();
        tx.send(Ok(Bytes::from_static(DONE.as_bytes()))).unwrap();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        let response = dispatch(
            client.post(&p.url),
            client.post(&d.url),
            "/v1/chat/completions",
            &HeaderMap::new(),
            pw.clone(),
            dw.clone(),
            slots.clone().acquire_owned().await.unwrap(),
        )
        .await
        .ok()
        .unwrap();
        let output = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        assert_eq!(output.as_ref(), format!("{TOKEN}{DONE}").as_bytes());
        eventually(|| slots.available_permits() == 1).await;
        assert_eq!(pw.load(), 0);
        assert_eq!(pw.circuit_breaker().total_failures(), 1);
        assert_eq!(dw.circuit_breaker().total_failures(), 0);
    }

    #[tokio::test]
    async fn pd_early_stream_decode_transport_failure_cancels_prefill() {
        let p = prefill_server(StatusCode::OK, Some(Arc::new(Notify::new())), None).await;
        let (d, tx) = decode_server().await;
        let pw = worker(&p.url);
        let dw = worker(&d.url);
        let slots = Arc::new(Semaphore::new(1));
        tx.send(Ok(Bytes::from_static(TOKEN.as_bytes()))).unwrap();
        let response = start(&p, &d, pw.clone(), dw.clone(), slots.clone())
            .await
            .ok()
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        assert!(body.next().await.unwrap().is_ok());
        tx.send(Err(io::Error::other("injected disconnect")))
            .unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .unwrap()
            .unwrap()
            .is_err());
        drop(body);
        eventually(|| slots.available_permits() == 1).await;
        assert_eq!(pw.load(), 0);
        assert_eq!(dw.load(), 0);
        assert_eq!(pw.circuit_breaker().total_failures(), 0);
        assert_eq!(dw.circuit_breaker().total_failures(), 1);
    }
}
