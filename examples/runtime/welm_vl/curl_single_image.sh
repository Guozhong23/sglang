#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to a small local JPEG or PNG image}"
PYTHON_BIN=${PYTHON_BIN:-python3}
SGLANG_URL=${SGLANG_URL:-http://127.0.0.1:6677}
request_file=$(mktemp)
response_file=$(mktemp)
trap 'rm -f -- "$request_file" "$response_file"' EXIT

# A data URL lets the server run on another machine without sharing image paths.
"${PYTHON_BIN}" - "${IMAGE_PATH}" "${SERVED_MODEL_NAME:-welmv45-vl}" "${MAX_TOKENS:-512}" > "${request_file}" <<'PY'
import base64
import json
import mimetypes
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
mime = mimetypes.guess_type(path.name)[0]
if mime not in ("image/jpeg", "image/png", "image/webp"):
    raise SystemExit("Use a JPEG, PNG or WebP image.")
image_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
json.dump({
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "请用中文简要描述这张图片中的主要内容。"},
    ]}],
    "temperature": 0,
    "max_tokens": int(sys.argv[3]),
    "stream": False,
}, sys.stdout, ensure_ascii=False)
PY

# Exactly one inference request; HTTP or malformed/empty responses fail the script.
curl --silent --show-error --fail-with-body \
    --connect-timeout 10 --max-time "${CURL_TIMEOUT:-600}" \
    "${SGLANG_URL%/}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@${request_file}" --output "${response_file}" || {
        cat "${response_file}"
        exit 1
    }
"${PYTHON_BIN}" - "${response_file}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    response = json.load(handle)
print(json.dumps(response, ensure_ascii=False, indent=2))
if response.get("error"):
    raise SystemExit("Server returned an error.")
choice = response["choices"][0]
message = choice["message"]
if not isinstance(message.get("content"), str) or not message["content"].strip():
    raise SystemExit("No final content returned; inspect reasoning_content, finish_reason and token budget.")
if choice.get("finish_reason") == "length":
    raise SystemExit("Generation reached the token limit; increase MAX_TOKENS and verify the complete answer.")
print("Single-image request completed. Check that the answer describes the supplied image.")
PY
