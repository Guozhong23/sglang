#!/usr/bin/env bash
# Profile exactly one image request on an already running WeLM-VL service.
# Use an idle, dedicated service: profiling and cache flushing affect all ranks.
set -euo pipefail

IMAGE_PATH=${IMAGE_PATH:-./dog.png}
SGLANG_URL=${SGLANG_URL:-http://127.0.0.1:7788}
SGLANG_URL=${SGLANG_URL%/}
PYTHON_BIN=${PYTHON_BIN:-python3}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-}
MAX_TOKENS=${MAX_TOKENS:-512}
PROMPT=${PROMPT:-请用中文简要描述这张图片中的主要内容。}
# This path belongs to the SERVER filesystem, even when the client is remote.
PROFILE_ROOT=${PROFILE_ROOT:-/data2/hw_lly/profiling3}
RUN_ID="welm_vl_single_$(date -u +%Y%m%dT%H%M%SZ)_$$"
PROFILE_DIR="${PROFILE_ROOT%/}/${RUN_ID}"
CLIENT_OUTPUT_DIR=${CLIENT_OUTPUT_DIR:-./welm_vl_profile_client/${RUN_ID}}
FLUSH_CACHE=${FLUSH_CACHE:-1}
FLUSH_TIMEOUT=${FLUSH_TIMEOUT:-30}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
PROFILE_TIMEOUT=${PROFILE_TIMEOUT:-1800}
WITH_STACK=${WITH_STACK:-0}
RECORD_SHAPES=${RECORD_SHAPES:-0}

for flag in FLUSH_CACHE WITH_STACK RECORD_SHAPES; do
    case "${!flag}" in 0|1) ;; *) echo "${flag} must be 0 or 1." >&2; exit 2 ;; esac
done
for setting in MAX_TOKENS FLUSH_TIMEOUT REQUEST_TIMEOUT PROFILE_TIMEOUT; do
    if [[ ! ${!setting} =~ ^[1-9][0-9]*$ ]]; then
        echo "${setting} must be a positive integer." >&2
        exit 2
    fi
done
if [[ ${PROFILE_ROOT} != /* ]]; then
    echo "PROFILE_ROOT must be an absolute path on the server." >&2
    exit 2
fi
if [[ ! -r ${IMAGE_PATH} || ! -s ${IMAGE_PATH} ]]; then
    echo "Image is missing, unreadable or empty: ${IMAGE_PATH}" >&2
    exit 2
fi
command -v curl >/dev/null
command -v "${PYTHON_BIN}" >/dev/null
mkdir -p -- "${CLIENT_OUTPUT_DIR}"

api_headers=()
admin_headers=()
if [[ -n ${SGLANG_API_KEY:-} ]]; then
    api_headers+=(-H "Authorization: Bearer ${SGLANG_API_KEY}")
fi
if [[ -n ${SGLANG_ADMIN_API_KEY:-${SGLANG_API_KEY:-}} ]]; then
    admin_headers+=(-H "Authorization: Bearer ${SGLANG_ADMIN_API_KEY:-${SGLANG_API_KEY:-}}")
fi

# Preserve raw bodies and HTTP status even for non-JSON control responses/errors.
# No retries: repeating start_profile or inference changes the capture boundary.
http() {
    local method=$1 endpoint=$2 name=$3 timeout=$4 body=${5:-}
    local headers=("${admin_headers[@]}")
    local request_args=()
    case "${endpoint}" in /v1/*) headers=("${api_headers[@]}") ;; esac
    if [[ -n ${body} ]]; then
        request_args+=(-H 'Content-Type: application/json' --data-binary "@${body}")
    fi
    if ! curl --silent --show-error --fail-with-body \
        --connect-timeout 10 --max-time "${timeout}" \
        --request "${method}" "${SGLANG_URL}${endpoint}" \
        "${headers[@]}" "${request_args[@]}" \
        --output "${CLIENT_OUTPUT_DIR}/${name}" \
        --write-out '%{http_code}\n' > "${CLIENT_OUTPUT_DIR}/${name}.http_status"; then
        echo "HTTP operation failed: ${method} ${endpoint}" >&2
        if [[ -f ${CLIENT_OUTPUT_DIR}/${name} ]]; then
            cat -- "${CLIENT_OUTPUT_DIR}/${name}" >&2
        fi
        return 1
    fi
    local status
    read -r status < "${CLIENT_OUTPUT_DIR}/${name}.http_status"
    if [[ ! ${status} =~ ^2[0-9][0-9]$ ]]; then
        echo "Unexpected HTTP ${status}: ${method} ${endpoint}; check the service URL." >&2
        return 1
    fi
}

profile_active=0
stop_profile() {
    # Mark before the call so a failed export is not automatically retried.
    profile_active=0
    echo "Stopping profiling; NPU trace export may take several minutes..."
    if ! http POST /stop_profile stop_profile.txt "${PROFILE_TIMEOUT}"; then
        echo "Stop/export failed. Inspect server logs and profiling state before another capture." >&2
        return 1
    fi
}
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ ${profile_active} == 1 ]]; then
        echo "Request failed or was interrupted; stopping this capture..." >&2
        if ! stop_profile; then status=1; fi
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

http GET /v1/models models.json 30
export IMAGE_PATH SERVED_MODEL_NAME MAX_TOKENS PROMPT PROFILE_DIR CLIENT_OUTPUT_DIR
export WITH_STACK RECORD_SHAPES SGLANG_URL FLUSH_CACHE
"${PYTHON_BIN}" - <<'PY'
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path

out = Path(os.environ["CLIENT_OUTPUT_DIR"])
models = json.loads((out / "models.json").read_text())["data"]
if not models:
    raise SystemExit("/v1/models returned no model.")
model = os.environ["SERVED_MODEL_NAME"] or models[0]["id"]
if model not in [item["id"] for item in models]:
    raise SystemExit(f"Model {model!r} is not served; check models.json.")
path = Path(os.environ["IMAGE_PATH"])
mime = mimetypes.guess_type(path.name)[0]
if mime not in ("image/jpeg", "image/png", "image/webp"):
    raise SystemExit("Use a JPEG, PNG or WebP image.")
data = path.read_bytes()
request = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {
            "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
        }},
        {"type": "text", "text": os.environ["PROMPT"]},
    ]}],
    "temperature": 0,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "stream": False,
}
profile = {
    "output_dir": os.environ["PROFILE_DIR"],
    # The current NPU backend maps the API's GPU activity to ProfilerActivity.NPU.
    "activities": ["CPU", "GPU"],
    "with_stack": os.environ["WITH_STACK"] == "1",
    "record_shapes": os.environ["RECORD_SHAPES"] == "1",
    "merge_profiles": False,
}
metadata = {
    "server_url": os.environ["SGLANG_URL"],
    "model": model,
    "image_path": str(path.resolve()),
    "image_sha256": hashlib.sha256(data).hexdigest(),
    "server_profile_dir": profile["output_dir"],
    "flush_kv_cache": os.environ["FLUSH_CACHE"] == "1",
    "vision_cache": "Not cleared by /flush_cache; ViT runs only on a cache miss.",
    "inference_requests": 1,
}
for name, obj in (("request.json", request), ("profile_request.json", profile), ("metadata.json", metadata)):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
print(f"Model discovered: {model}")
PY

echo "Client request/response files: ${CLIENT_OUTPUT_DIR}"
echo "Server profiling directory: ${PROFILE_DIR}"
echo "Run with no other inference or profiling sessions; this script sends one inference request."
echo "Note: flushing KV cache does not flush vision embeddings. Reused images may skip ViT."
if [[ ${FLUSH_CACHE} == 1 ]]; then
    http POST "/flush_cache?timeout=${FLUSH_TIMEOUT}" flush_cache.txt "$((FLUSH_TIMEOUT + 30))"
fi

if ! http POST /start_profile start_profile.txt "${PROFILE_TIMEOUT}" "${CLIENT_OUTPUT_DIR}/profile_request.json"; then
    echo "Start failed; no inference was sent. If a timeout or partial start occurred, inspect the server and stop profiling there before retrying." >&2
    exit 1
fi
profile_active=1
http POST /v1/chat/completions response.json "${REQUEST_TIMEOUT}" "${CLIENT_OUTPUT_DIR}/request.json"
stop_profile

# Validate only after stop so malformed responses still leave profiling stopped.
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["CLIENT_OUTPUT_DIR"])
response = json.loads((out / "response.json").read_text())
if response.get("error"):
    raise SystemExit(f"Inference failed: {response['error']}")
choice = response["choices"][0]
message = choice["message"]
usage = response.get("usage", {})
summary = {
    "response_id": response.get("id"),
    "finish_reason": choice.get("finish_reason"),
    "usage": usage,
    "server_profile_dir": os.environ["PROFILE_DIR"],
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not (message.get("content") or message.get("reasoning_content")):
    raise SystemExit("Response contains neither content nor reasoning; inspect response.json.")
if choice.get("finish_reason") == "length":
    print("Generation reached MAX_TOKENS; this capture includes a truncated answer.")
cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
if os.environ["FLUSH_CACHE"] == "1" and cached:
    print(f"WARNING: {cached} prompt tokens were cached despite flushing; check for concurrent traffic.")
print("One request captured; inspect the server trace files to confirm device events and ViT coverage.")
PY

echo "NPU traces are on the SERVER under: ${PROFILE_DIR}"
echo "Look recursively for trace_view.json / operator and kernel reports; filenames depend on torch_npu."
