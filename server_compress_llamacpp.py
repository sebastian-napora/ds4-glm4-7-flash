#!/usr/bin/env python3
"""LiteLLM proxy for the llama.cpp GLM backend.

Architecture:
    VS Code / GitHub Copilot plugin -> LiteLLM (12111) -> glm-xml-proxy (12180) -> llama.cpp (12112)

The proxy at port 12180 intercepts llama.cpp's streaming response and converts
GLM's native <tool_call>...</tool_call> XML format to OpenAI's tool_calls JSON format.

llama.cpp itself is untouched — it's a drop-in replacement for any OpenAI-compatible server.
"""

import asyncio
import json
import logging
import os
import re
import signal
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import AsyncIterator, Iterator
from urllib.parse import urlencode

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

# ── Logging setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "glm_xml_proxy.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("glm_xml_proxy")

# ── Tool allowlist ────────────────────────────────────────────────────────────
# Only these tools are forwarded to llama.cpp.
# A tool is allowed if it appears in ALLOWED_TOOLS (or TOOL_BLOCKLIST as a safety net).
# All other tools from GitHub Copilot are filtered out.
ALLOWED_TOOLS: frozenset[str] = frozenset({
    # Model's own tools
    "grep_search",
    "glob",
    "read_file",
    "read_multiple_files",
    "write_to_file",
    "str_replace_editor",
    "notebook_edit",
    # GitHub Copilot's tools — also allowed so the model can see them
    "createDirectory",
    "createFile",
    "editFiles",
    "deleteFile",
    "deleteDirectory",
    "listDirectory",
    "moveFile",
    "searchReplace",
    "read",
    "write",
})

TOOL_BLOCKLIST: frozenset[str] = frozenset({})  # no hard blocks — all Copilot tools pass through

def filter_request_tools(body: bytes) -> bytes:
    """Remove Copilot-specific tools from the request body before forwarding to llama.cpp."""
    try:
        obj = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body

    tools = obj.get("tools")
    if not tools:
        return body

    filtered = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        if name in TOOL_BLOCKLIST:
            logger.info("Filtering out blocked tool: %s", name)
            continue
        if name not in ALLOWED_TOOLS:
            logger.info("Filtering out non-allowed tool: %s", name)
            continue
        filtered.append(tool)

    obj["tools"] = filtered if filtered else None
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ── Tool call accumulator ─────────────────────────────────────────────────────
# llama.cpp emits delta.tool_calls token-by-token in OpenAI format (NOT XML).
# We accumulate streamed fragments and emit the complete tool_calls at the end.


class ToolCallAccumulator:
    """Accumulates streamed tool_calls + content across SSE chunks.

    llama.cpp streams function arguments token-by-token across multiple chunks.
    This class merges all fragments and emits a final complete chunk when
    finish_reason="tool_calls" is seen, or flushes on StreamClosed.
    """

    def __init__(self):
        self._tc_fragments: list[dict] = []   # partial tool_calls being built
        self._content: list[str] = []         # accumulated text content
        self._base: dict = {}                 # id, created, model
        self._seen_finish = False

    def ingest(self, obj: dict) -> list[dict]:
        """Process a parsed SSE chunk JSON object. Returns output chunk dicts."""
        results = []

        choices = obj.get("choices", [])
        if not choices:
            return results

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Capture base info from first chunk
        if not self._base:
            self._base = {
                "id": obj.get("id", "chatcmpl-fix"),
                "created": obj.get("created", 0),
                "model": obj.get("model", ""),
            }

        # Accumulate content
        content = delta.get("content")
        if content:
            self._content.append(content)

        # Merge streamed tool_calls fragments
        tc_delta = delta.get("tool_calls", [])
        if tc_delta:
            self._merge_tc_delta(tc_delta)

        # On finish_reason=tool_calls: emit complete tool_calls BEFORE the original chunk.
        # Use the CURRENT chunk's metadata (created, id) for correct timestamps.
        if finish_reason == "tool_calls" and self._tc_fragments:
            self._seen_finish = True
            results.append(self._build_final_chunk(
                chunk_id=obj.get("id", self._base.get("id", "chatcmpl-fix")),
                chunk_created=obj.get("created", self._base.get("created", 0)),
            ))

        return results

    def _merge_tc_delta(self, tc_delta: list):
        """Merge a streamed tool_calls delta into accumulated fragments."""
        for tc in tc_delta:
            idx = tc.get("index", 0)
            while len(self._tc_fragments) <= idx:
                self._tc_fragments.append({
                    "index": len(self._tc_fragments),
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })

            frag = self._tc_fragments[idx]
            if tc.get("id") and not frag["id"]:
                frag["id"] = tc["id"]
            if tc.get("type"):
                frag["type"] = tc["type"]

            fn = tc.get("function", {})
            frag["function"]["name"] += fn.get("name", "")
            frag["function"]["arguments"] += fn.get("arguments", "")

    def _build_final_chunk(self, chunk_id: str = "", chunk_created: int = 0) -> dict:
        """Build a chunk with the complete accumulated tool_calls."""
        content_str = "".join(self._content)
        return {
            "id": chunk_id or self._base.get("id", "chatcmpl-fix"),
            "object": "chat.completion.chunk",
            "created": chunk_created or self._base.get("created", 0),
            "model": self._base.get("model", ""),
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content_str if content_str else None,
                    "tool_calls": self._tc_fragments,
                },
                "finish_reason": "tool_calls",
            }],
        }

    def flush(self) -> list[dict]:
        """Flush pending accumulated data (called on StreamClosed)."""
        results = []
        if self._tc_fragments and not self._seen_finish:
            results.append(self._build_final_chunk())
            self._seen_finish = True
        return results

    @property
    def has_tool_calls(self) -> bool:
        return bool(self._tc_fragments)


# ── HTTP Proxy (for streaming responses) ──────────────────────────────────────

LLAMA_PORT = int(os.environ.get("LLAMA_BACKEND_PORT", "12112"))
PROXY_PORT = int(os.environ.get("XML_PROXY_PORT", "12180"))
LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")

LLAMA_BASE = f"http://{LLAMA_HOST}:{LLAMA_PORT}"


async def proxy_chat(request: Request) -> Response:
    """Proxy /v1/chat/completions, transforming GLM XML tool_calls to OpenAI format."""
    body = await request.body()
    try:
        pre_count = len(json.loads(body).get("tools", []))
    except Exception:
        pre_count = -1
    body = filter_request_tools(body)  # strip Copilot-only tools
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)  # recalculate from actual (smaller) body

    # Log how many tools remain after filtering
    try:
        tools = json.loads(body).get("tools", [])
        logger.info("Tools after filter: %d (was %d)", len(tools), pre_count)
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST",
                f"{LLAMA_BASE}/v1/chat/completions",
                content=body,
                headers=headers,
            ) as llama_resp:

                is_streaming = "text/event-stream" in llama_resp.headers.get("content-type", "")

                # Always read the full response body first to avoid httpx stream issues
                full_body = await llama_resp.aread()

                if not is_streaming:
                    # Non-streaming — return as-is
                    return Response(
                        content=full_body,
                        status_code=llama_resp.status_code,
                        headers=dict(llama_resp.headers),
                    )

                # Streaming: parse, transform, return as a new stream from memory
                acc = ToolCallAccumulator()
                output_lines: list[bytes] = []

                for raw_line in full_body.split(b"\n"):
                    raw_line = raw_line.rstrip(b"\r")
                    if not raw_line:
                        continue

                    # Emit any pre-generated chunks from accumulator first
                    for pre_chunk in _parse_and_ingest(raw_line, acc):
                        output_lines.append(pre_chunk + b"\n")

                    # Pass through the original line
                    output_lines.append(raw_line + b"\n")

                # If stream was truncated, flush any pending tool_calls
                if acc.has_tool_calls and not acc._seen_finish:
                    for flush_chunk in acc.flush():
                        output_lines.append(b"data: " + json.dumps(flush_chunk, ensure_ascii=False).encode() + b"\n")

                if not output_lines:
                    return Response(content=b"", status_code=200, media_type="text/event-stream")

                async def memory_stream():
                    for line in output_lines:
                        yield line

                return StreamingResponse(
                    memory_stream(),
                    status_code=llama_resp.status_code,
                    media_type="text/event-stream",
                )

    except httpx.HTTPError as e:
        logger.error("Upstream llama.cpp error: %s", e)
        return JSONResponse({"error": {"message": str(e), "type": "upstream_error"}}, status_code=502)


def _parse_and_ingest(line: bytes, acc: ToolCallAccumulator) -> list[bytes]:
    """Parse a single SSE line, feed to accumulator, return pre-generated chunks as bytes."""
    if not line.startswith(b"data: "):
        return []

    data = line[6:].strip()
    if data in (b"[DONE]", b""):
        return []

    try:
        obj = json.loads(data.decode("utf-8"))
        results = acc.ingest(obj)
        return [b"data: " + json.dumps(c, ensure_ascii=False).encode() for c in results]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    except httpx.HTTPError as e:
        logger.error("Upstream llama.cpp error: %s", e)
        return JSONResponse({"error": {"message": str(e), "type": "upstream_error"}}, status_code=502)


async def proxy_models(request: Request) -> Response:
    """Proxy /v1/models — pass through."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{LLAMA_BASE}/v1/models", timeout=10.0)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={"content-type": "application/json"},
            )
    except httpx.HTTPError as e:
        return JSONResponse({"error": {"message": str(e), "type": "upstream_error"}}, status_code=502)


async def proxy_health(request: Request) -> Response:
    """Proxy /health."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{LLAMA_BASE}/health", timeout=5.0)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={"content-type": "application/json"},
            )
    except httpx.HTTPError:
        return JSONResponse({"status": "ok", "xml_proxy": "active"}, status_code=200)


# ── LiteLLM upstream ───────────────────────────────────────────────────────────

import litellm

import glm_compress       # noqa: F401 — strips thinking tokens from history
import glm_token_tracker  # noqa: F401 — records per-request token usage

glm_compress.register()
glm_token_tracker.register()

litellm_logger = logging.getLogger("litellm.image_request")
litellm_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_DIR / "litellm_llamacpp_image_requests.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(asctime)s %(name)-25s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(ch)
litellm_logger.info("=" * 60)
litellm_logger.info("LiteLLM llama.cpp proxy started")

os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_REQUEST_LOGGING"] = "true"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "litellm_llamacpp_detailed.log"),
        logging.StreamHandler(),
    ],
)
logging.getLogger("litellm").setLevel(logging.DEBUG)
logger = logging.getLogger("server_compress_llamacpp")

LITELLM_PORT = os.environ.get("LITE_LLM_PROXY_PORT", "12111")
LITELLM_HOST = os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0")

# Point LiteLLM at the XML-transforming proxy instead of direct llama.cpp
XML_PROXY_BASE = os.environ.get("XML_PROXY_BASE", f"http://127.0.0.1:{PROXY_PORT}")

BASE_CONFIG_PATH = Path(__file__).parent / "lite_llm_config_llamacpp.yaml"
CONFIG_PATH = LOG_DIR / f"lite_llm_config_llamacpp.{LLAMA_PORT}.runtime.yaml"

config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")
# Replace the backend URL in config to point to the XML transform proxy instead of direct llama.cpp
config_text = config_text.replace(
    f"http://localhost:{LLAMA_PORT}/v1",
    f"http://127.0.0.1:{PROXY_PORT}/v1"
)
CONFIG_PATH.write_text(config_text, encoding="utf-8")

logger.info("XML transform proxy: %s", XML_PROXY_BASE)
logger.info("LiteLLM config: %s", CONFIG_PATH)

os.environ.pop("LITELLM_MASTER_KEY", None)
os.environ.pop("LITELLM_SALT_KEY", None)
os.environ["CONFIG_FILE_PATH"] = str(CONFIG_PATH)


# ── App startup ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting GLM XML tool_call proxy on :%d -> llama.cpp :%d", PROXY_PORT, LLAMA_PORT)
    logger.info("LiteLLM will use XML proxy as backend: %s/v1", XML_PROXY_BASE)
    logger.info("=" * 60)

    # Build the XML transform proxy app
    async def root_handler(request: Request) -> Response:
        return HTMLResponse("<h1>GLM XML Proxy</h1>")

    xml_proxy_app = Starlette(
        routes=[
            Route("/v1/chat/completions", proxy_chat, methods=["POST"]),
            Route("/v1/models", proxy_models, methods=["GET"]),
            Route("/health", proxy_health, methods=["GET"]),
            Route("/", root_handler, methods=["GET"]),
        ],
        middleware=[
            (CORSMiddleware, [], {"allow_origins": ["*"], "allow_methods": ["*"], "allow_headers": ["*"]}),
        ],
    )

    xml_proxy = uvicorn.Config(
        xml_proxy_app,
        host="0.0.0.0",
        port=PROXY_PORT,
        log_level="info",
    )
    xml_proxy_runner = uvicorn.Server(xml_proxy)

    # Start XML proxy in background thread
    def run_xml_proxy():
        xml_proxy_runner.run()

    xml_thread = threading.Thread(target=run_xml_proxy, daemon=True)
    xml_thread.start()
    logger.info("XML proxy started on port %d", PROXY_PORT)

    # Give XML proxy a moment to bind
    import time; time.sleep(1)

    # Build LiteLLM app with CORS middleware for cross-device access
    from litellm.proxy.proxy_server import app as litellm_app

    app = CORSMiddleware(
        litellm_app,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    logger.info("Starting LiteLLM on %s:%s", LITELLM_HOST, LITELLM_PORT)
    uvicorn.run(
        app,
        host=LITELLM_HOST,
        port=int(LITELLM_PORT),
        reload=False,
        log_level="info",
    )