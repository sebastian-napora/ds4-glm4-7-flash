#!/usr/bin/env python3
"""
LiteLLM proxy for llama.cpp GLM backend — with GLM XML → OpenAI tool_calls transform.

Architecture:
    VS Code / GitHub Copilot plugin -> LiteLLM (12111) -> [XML transform proxy (12180)] -> llama-server (12112)

GLM-4.7-Flash generates <tool_call>...</tool_call> XML format (especially with
many tools like Copilot's). This module starts two services:
  1. XML transform proxy on 12180 — intercepts streaming output from llama-server,
     converts GLM XML tool_calls to OpenAI JSON tool_calls.
  2. LiteLLM proxy on 12111 — serves Copilot, routing to the XML proxy.

By default the proxy describes tools in the prompt and removes the native OpenAI
"tools" field before forwarding to llama.cpp. That keeps llama.cpp from trying
to parse GLM's XML tool syntax itself; the proxy owns XML parsing instead.

Both run in the foreground so all output streams to the terminal.
"""

import asyncio
import html
import json
import logging
import os
import random
import re
import threading
import time as _time
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_log = logging.getLogger("glm-xml-proxy")  # used throughout

# ── GLM XML tool_calls extractor ────────────────────────────────────────────────

class GLMToolCallExtractor:
    """Accumulate streamed text and extract complete XML tool-call blocks.

    GLM-4.7-Flash outputs tool_calls as XML fragments across multiple SSE chunks.
    This extractor buffers text, detects complete blocks, converts them to OpenAI
    tool_calls format, and strips the XML from the streamed content delta.
    """

    def __init__(self, allowed_tool_names: set[str] | None = None):
        self._buffer = ""
        self._tool_calls: list[dict] = []
        self._content_parts: list[str] = []
        self._base: dict = {}       # id / created / model from first chunk
        self._seen_finish = False
        self._allowed_tool_names = allowed_tool_names or set()

    def ingest_text(self, text: str) -> list[tuple[str, str | dict]]:
        """Feed raw text delta. Returns list of (event_type, data) to emit."""
        results: list[tuple[str, str | dict]] = []
        self._buffer += text

        while self._buffer:
            call_start = self._find_next_call_start()

            if call_start is None:
                # No complete tag start, but the buffer may end with a partial
                # XML prefix split across SSE chunks. Hold that suffix back.
                keep = self._partial_call_prefix_len()
                emit = self._buffer[:-keep] if keep else self._buffer
                if emit.strip():
                    self._content_parts.append(emit)
                    results.append(("content", emit))
                self._buffer = self._buffer[len(self._buffer) - keep:] if keep else ""
                break

            start, tag_name, open_tag_len = call_start

            # Emit content before this tool-call tag
            if start > 0:
                self._content_parts.append(self._buffer[:start])
                results.append(("content", self._buffer[:start]))
                self._buffer = self._buffer[start:]

            end_tag = f"</{tag_name}>"

            # Find the closing tag (buffer still starts with opening tag)
            end = self._buffer.find(end_tag)
            if end < 0:
                # Incomplete block — keep entire tagged buffer and wait for more
                break

            block = self._buffer[open_tag_len:end]
            self._buffer = self._buffer[end + len(end_tag):]

            if tag_name == "tool_call":
                tc = self._parse_tool_call(block)
            else:
                tc = self._parse_direct_tool_call(tag_name, block)
            if tc:
                self._tool_calls.append(tc)
                results.append(("tool_call", tc))

        return results

    def _find_next_call_start(self) -> tuple[int, str, int] | None:
        """Return (start_index, tag_name, opening_tag_length) for next call tag."""
        matches: list[tuple[int, str, int]] = []

        normalized_start = self._buffer.find("<tool_call>")
        if normalized_start >= 0:
            matches.append((normalized_start, "tool_call", len("<tool_call>")))

        for tool_name in self._allowed_tool_names:
            if not tool_name or tool_name == "tool_call":
                continue
            match = re.search(
                rf"<{re.escape(tool_name)}(?:\s[^>]*)?>",
                self._buffer,
                re.DOTALL,
            )
            if match:
                matches.append((match.start(), tool_name, match.end() - match.start()))

        if not matches:
            return None
        return min(matches, key=lambda item: item[0])

    def _partial_call_prefix_len(self) -> int:
        """How many trailing chars to keep because they may start a tool tag."""
        candidates = ["<tool_call>"]
        candidates.extend(f"<{name}" for name in self._allowed_tool_names if name)

        keep = 0
        for candidate in candidates:
            max_len = min(len(candidate) - 1, len(self._buffer))
            for plen in range(max_len, 0, -1):
                if self._buffer.endswith(candidate[:plen]):
                    keep = max(keep, plen)
                    break
        return keep

    def _parse_tool_call(self, xml_block: str) -> dict | None:
        """Parse the inner text of a <tool_call>...</tool_call> block (tags already
        stripped by ingest_text) into OpenAI tool_calls format."""
        # Extract tool name: everything before the first tag (or whole block if no args)
        name_match = re.match(r"^\s*([^<\s][^<]*?)\s*(?:<|$)", xml_block, re.DOTALL)
        if not name_match:
            _log.warning("Cannot parse tool_call name from: %s", xml_block[:120])
            return None

        tool_name = self._resolve_tool_name(name_match.group(1).strip())

        # Extract arguments: <arg_key>key</arg_key><arg_value>val</arg_value>
        # arg values can span multiple lines, hence DOTALL.
        args: dict = {}
        for key_match in re.finditer(r"<arg_key>([^<]*)</arg_key>", xml_block):
            key = key_match.group(1).strip()
            after_key = xml_block[key_match.end():]
            val_match = re.search(r"<arg_value>(.*?)</arg_value>", after_key, re.DOTALL)
            raw_val = val_match.group(1) if val_match else ""
            # GLM emits all values as text; recover JSON types (numbers, bools,
            # objects, arrays) when the value parses as JSON, else keep string.
            stripped = raw_val.strip()
            if stripped and stripped[0] in "{[0123456789-tfn":
                try:
                    args[key] = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    args[key] = raw_val
            else:
                args[key] = raw_val

        tc_id = f"call_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=24))}"
        return {
            "index": len(self._tool_calls),
            "id": tc_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def _parse_direct_tool_call(self, tool_name: str, xml_block: str) -> dict | None:
        """Parse leaked direct tool tags, e.g. <runSubagent><prompt>...</prompt>."""
        tool_name = self._resolve_tool_name(tool_name)
        args: dict = {}
        for match in re.finditer(
            r"<([A-Za-z_][A-Za-z0-9_.:-]*)\b[^>]*>(.*?)</\1>",
            xml_block,
            re.DOTALL,
        ):
            key = match.group(1).strip()
            raw_val = html.unescape(match.group(2).strip())
            stripped = raw_val.strip()
            if stripped and stripped[0] in "{[0123456789-tfn":
                try:
                    args[key] = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    args[key] = raw_val
            else:
                args[key] = raw_val

        if not args:
            _log.warning("Cannot parse direct tool_call args for %s: %s", tool_name, xml_block[:120])
            return None

        tc_id = f"call_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=24))}"
        return {
            "index": len(self._tool_calls),
            "id": tc_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def _resolve_tool_name(self, tool_name: str) -> str:
        """Map common model-emitted aliases back to the provided OpenAI tool name."""
        if not self._allowed_tool_names or tool_name in self._allowed_tool_names:
            return tool_name

        aliases = [
            tool_name.replace("/", "_"),
            re.sub(r"[^A-Za-z0-9_-]", "_", tool_name),
        ]
        for alias in aliases:
            if alias in self._allowed_tool_names:
                return alias

        lower_map = {name.lower(): name for name in self._allowed_tool_names}
        for alias in aliases:
            resolved = lower_map.get(alias.lower())
            if resolved:
                return resolved

        _log.warning("Tool name %r not found in provided tools; forwarding as emitted", tool_name)
        return tool_name

    def build_final_chunk(self, chunk_id: str = "", chunk_created: int = 0) -> dict:
        """Build the final completion chunk with accumulated tool_calls."""
        content_str = "".join(self._content_parts)
        return {
            "id": chunk_id or self._base.get("id", "chatcmpl-glm"),
            "object": "chat.completion.chunk",
            "created": chunk_created or self._base.get("created", 0),
            "model": self._base.get("model", "glm-4.7-flash-llamacpp"),
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content_str if content_str else None,
                    "tool_calls": self._tool_calls if self._tool_calls else None,
                },
                "finish_reason": "tool_calls" if self._tool_calls else "stop",
            }],
        }

    @property
    def has_tool_calls(self) -> bool:
        return bool(self._tool_calls)


def _parse_sse_line(raw: str) -> dict | None:
    """Parse a single SSE data: line into a JSON dict."""
    if not raw.startswith("data: "):
        return None
    data = raw[6:].strip()
    if data in ("[DONE]", ""):
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _chunk_to_line(chunk: dict) -> bytes:
    # SSE events MUST be terminated by a blank line ("\n\n"); a single "\n"
    # makes downstream SSE parsers (e.g. LiteLLM) buffer forever and yield nothing.
    return b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"


def _message_tool_call(tool_call: dict) -> dict:
    """OpenAI non-streaming message.tool_calls do not use the streaming index key."""
    return {
        "id": tool_call.get("id"),
        "type": tool_call.get("type", "function"),
        "function": tool_call.get("function", {}),
    }


def _transform_non_streaming_completion(content: bytes, allowed_tool_names: set[str]) -> bytes | None:
    """Convert XML tool calls inside a non-streaming chat completion response."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    changed = False
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        content_text = message.get("content")
        if not isinstance(content_text, str) or "<" not in content_text:
            continue

        extractor = GLMToolCallExtractor(allowed_tool_names=allowed_tool_names)
        extractor.ingest_text(content_text)
        if not extractor.has_tool_calls:
            continue

        message["content"] = None
        message["tool_calls"] = [_message_tool_call(tc) for tc in extractor._tool_calls]
        choice["finish_reason"] = "tool_calls"
        changed = True

    if not changed:
        return None

    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ════════════════════════════════════════════════════════════════════════════════
# XML TRANSFORM PROXY  (runs on PROXY_PORT → llama-server)
# ════════════════════════════════════════════════════════════════════════════════

LLAMA_PORT    = int(os.environ.get("LLAMA_BACKEND_PORT", "12112"))
LLAMA_HOST    = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_BASE    = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
PROXY_PORT    = int(os.environ.get("XML_PROXY_PORT", "12180"))


# ── GBNF grammar fixer ──────────────────────────────────────────────────────────
# llama.cpp's GBNF grammar parser does not support \d, \w, \s etc.
# Many MCP tools (e.g. Figma) ship grammars that use \d for digits.
# Replace these with equivalent character classes in any string value
# inside the request body before forwarding to llama-server.
#
# Note: GBNF grammar strings are embedded in the JSON body inside
# tool definitions (function.parameters.grammar or similar fields).


# ── Simple 4-tool set for testing ──────────────────────────────────────────────
# Replace whatever tools Copilot sends with this minimal set.
# Set SIMPLE_TOOLS=False to restore the original tool passthrough.


SIMPLE_TOOLS_ENABLED = False  # flip to True to replace client tools with the simple test set
PROMPT_TOOLS_AS_XML = os.environ.get("GLM_PROMPT_TOOLS_AS_XML", "true").lower() not in {
    "0", "false", "no", "off",
}

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file from the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based). Default: 0.",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Default: all lines.",
                        "default": None,
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, replacing all existing content. "
                          "Use this to create new files or overwrite existing ones entirely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path of the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing a specific substring (old_string) "
                          "with new content (new_string). The old_string MUST match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace. "
                                      "Must match verbatim — no regex.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file from the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path of the file to delete.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
]


def _replace_tools_with_simple(body_str: str) -> tuple[str, int]:
    """Replace the tools array with our 4-tool set.

    Returns (fixed_body, original_tool_count).
    """
    try:
        payload = json.loads(body_str)
    except (json.JSONDecodeError, TypeError):
        return body_str, 0

    original_tools = payload.get("tools", [])
    original_count = len(original_tools)

    if SIMPLE_TOOLS_ENABLED:
        payload["tools"] = TOOL_DEFINITIONS
        if original_count > 4:
            _log.warning(
                "Tool replacement: reduced %d tools → %d  (SIMPLE_TOOLS_ENABLED=True)",
                original_count, len(TOOL_DEFINITIONS),
            )

    return json.dumps(payload, ensure_ascii=False), original_count


def _tool_prompt(tools: list[dict]) -> str:
    """Render OpenAI tool definitions as compact prompt instructions for GLM."""
    rendered_tools: list[str] = []
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        desc = (fn.get("description") or "").strip()
        params = fn.get("parameters") or {}
        params_json = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
        if desc:
            rendered_tools.append(f"- {name}: {desc}\n  parameters: {params_json}")
        else:
            rendered_tools.append(f"- {name}\n  parameters: {params_json}")

    return (
        "You have access to tools. When a tool is needed, output exactly one "
        "tool call in this XML format and no surrounding markdown:\n"
        "<tool_call>tool_name"
        "<arg_key>argument_name</arg_key><arg_value>argument_value</arg_value>"
        "</tool_call>\n"
        "Use one <arg_key>/<arg_value> pair for each argument. Use the exact tool "
        "name and argument names from the available tools below. For booleans, "
        "numbers, arrays, or objects, put valid JSON in <arg_value>. Do not output "
        "direct tool-name XML tags such as <runSubagent> or internal planning "
        "wrappers; convert every tool action to the <tool_call> format above.\n\n"
        "Available tools:\n"
        + "\n".join(rendered_tools)
    )


def _extract_tool_names(body_str: str) -> set[str]:
    """Return OpenAI function tool names from a request body."""
    try:
        payload = json.loads(body_str)
    except (json.JSONDecodeError, TypeError):
        return set()

    names: set[str] = set()
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {})
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _prompt_tools_and_disable_native(body_str: str) -> tuple[str, int, bool]:
    """Move OpenAI tool definitions into a system prompt and remove native tools.

    llama.cpp can try to parse model-emitted tool calls before our transform proxy
    sees them. Prompting the tools while forwarding a plain chat request avoids
    that native parser path and lets this proxy convert GLM XML to OpenAI JSON.
    """
    if not PROMPT_TOOLS_AS_XML:
        return body_str, 0, False

    try:
        payload = json.loads(body_str)
    except (json.JSONDecodeError, TypeError):
        return body_str, 0, False

    tools = payload.get("tools") or []
    if not tools:
        return body_str, 0, False

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return body_str, len(tools), False

    prompt = _tool_prompt(tools)
    tool_msg = {
        "role": "system",
        "content": prompt,
    }

    insert_at = 0
    while (
        insert_at < len(messages)
        and isinstance(messages[insert_at], dict)
        and messages[insert_at].get("role") == "system"
    ):
        insert_at += 1
    messages.insert(insert_at, tool_msg)

    payload["messages"] = messages
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload.pop("parallel_tool_calls", None)

    return json.dumps(payload, ensure_ascii=False), len(tools), True


def _fix_grammar_strings(body_str: str) -> tuple[str, int]:
    """Replace regex-style escapes with GBNF-compatible char-classes in JSON body.

    Returns (fixed_body, num_fixes_applied).
    """
    try:
        payload = json.loads(body_str)
    except (json.JSONDecodeError, TypeError):
        return body_str, 0

    num_fixes = 0

    def fix_string(s: str) -> str:
        nonlocal num_fixes
        # Map regex escapes that GBNF doesn't support → GBNF character classes.
        # Order matters: replace \b and \B first (word-boundary), then chars.
        fixes = [
            # word/non-word boundary — GBNF has no boundary concept; strip quantifier
            (r'\b', ''),
            (r'\B', ''),
            # digit/non-digit
            (r'\\d([+*?]?)', r'[0-9]\1'),
            (r'\\D([+*?]?)', r'[^0-9]\1'),
            # whitespace/non-whitespace
            (r'\\s([+*?]?)', r'[ \t\n\r\f\v]\1'),
            (r'\\S([+*?]?)', r'[^ \t\n\r\f\v]\1'),
            # word/non-word character
            (r'\\w([+*?]?)', r'[a-zA-Z0-9_]\1'),
            (r'\\W([+*?]?)', r'[^a-zA-Z0-9_]\1'),
            # literal backslash+digit that somehow survived
            (r'\\1', ''), (r'\\2', ''), (r'\\3', ''),
        ]
        result = s
        for pattern, replacement in fixes:
            new_result = re.sub(pattern, replacement, result)
            if new_result != result:
                num_fixes += 1
                result = new_result
        return result

    def walk(obj):
        if isinstance(obj, str):
            return fix_string(obj)
        elif isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [walk(item) for item in obj]
        return obj

    if isinstance(payload.get("messages"), list):
        payload["messages"] = [walk(msg) for msg in payload["messages"]]
    if isinstance(payload.get("tools"), list):
        payload["tools"] = [walk(tool) for tool in payload["tools"]]

    if num_fixes > 0:
        return json.dumps(payload, ensure_ascii=False), num_fixes
    return body_str, 0


async def proxy_chat(request: Request) -> Response:
    """Proxy /v1/chat/completions, transforming GLM XML → OpenAI tool_calls."""
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # Step 1: Fix GBNF grammar (regex escapes → GBNF char-classes)
    body_str = body.decode("utf-8", errors="replace")
    fixed_body_str, num_grammar_fixes = _fix_grammar_strings(body_str)

    # Step 2: Replace tools with our simple 4-tool set (if enabled)
    replaced_body_str, orig_tool_count = _replace_tools_with_simple(fixed_body_str)
    tool_names = _extract_tool_names(replaced_body_str)

    # Step 3: Prompt tools as XML instructions, then remove native tool schemas so
    # llama.cpp does not attempt to parse GLM's XML tool-call text itself.
    forwarded_body_str, prompted_tool_count, prompted_tools = _prompt_tools_and_disable_native(
        replaced_body_str,
    )

    body = forwarded_body_str.encode("utf-8")

    # Debug: log request overview
    try:
        req_payload = json.loads(forwarded_body_str)
        n_tools = len(req_payload.get("tools", []))
        n_msgs = len(req_payload.get("messages", []))
        body_kb = len(body) / 1024
        _log.debug(
            "proxy_chat  native_tools=%d(%s)  prompted_tools=%d  msgs=%d  "
            "body=%.1fKB  grammar_fixes=%d  orig_tools=%d",
            n_tools, "simple" if SIMPLE_TOOLS_ENABLED else "original",
            prompted_tool_count if prompted_tools else 0,
            n_msgs, body_kb, num_grammar_fixes, orig_tool_count,
        )
    except Exception:
        _log.debug(
            "proxy_chat  body_bytes=%d  grammar_fixes=%d  orig_tools=%d  "
            "prompted_tools=%d",
            len(body), num_grammar_fixes, orig_tool_count,
            prompted_tool_count if prompted_tools else 0,
        )

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST",
                f"{LLAMA_BASE}/v1/chat/completions",
                content=body,
                headers=headers,
            ) as llama_resp:
                is_stream = "text/event-stream" in llama_resp.headers.get("content-type", "")
                _log.debug(
                    "llama-server resp  status=%d  is_stream=%s  content-type=%s",
                    llama_resp.status_code, is_stream,
                    llama_resp.headers.get("content-type", ""),
                )

                if not is_stream:
                    content = await llama_resp.aread()
                    transformed = _transform_non_streaming_completion(content, tool_names)
                    if transformed is not None:
                        _log.info("non-streaming resp  transformed XML tool_call to OpenAI tool_calls")
                        content = transformed
                    else:
                        _log.debug("non-streaming resp  bytes=%d", len(content))
                    headers_out = dict(llama_resp.headers)
                    headers_out.pop("content-length", None)
                    headers_out["content-type"] = "application/json"
                    return Response(
                        content=content,
                        status_code=llama_resp.status_code,
                        headers=headers_out,
                    )

                # ── Streaming transform ────────────────────────────────────────
                acc = GLMToolCallExtractor(allowed_tool_names=tool_names)
                output_lines: list[bytes] = []
                native_tool_calls = False

                def passthrough(line: str) -> None:
                    # SSE events must end with a blank line ("\n\n").
                    output_lines.append(line.encode() + b"\n\n")

                def finish_chunk(reason: str, chunk_id: str = "", created: int = 0) -> dict:
                    return {
                        "id": chunk_id or acc._base.get("id", "chatcmpl-glm"),
                        "object": "chat.completion.chunk",
                        "created": created or acc._base.get("created", 0),
                        "model": acc._base.get("model", "glm-4.7-flash-llamacpp"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                    }

                async for raw_line in llama_resp.aiter_lines():
                    raw_line = raw_line.rstrip("\r")
                    if not raw_line:
                        continue

                    parsed = _parse_sse_line(raw_line)
                    if parsed is None:
                        passthrough(raw_line)
                        continue

                    choices = parsed.get("choices", [])
                    if not choices:
                        passthrough(raw_line)
                        continue

                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")

                    # Capture base info from first chunk
                    if not acc._base:
                        acc._base = {
                            "id": parsed.get("id", "chatcmpl-glm"),
                            "created": parsed.get("created", 0),
                            "model": parsed.get("model", "glm-4.7-flash-llamacpp"),
                        }

                    raw_text = delta.get("content", "") or ""
                    tc_deltas = delta.get("tool_calls", [])

                    if tc_deltas:
                        # llama-server already emitting OpenAI tool_calls — pass through.
                        # But also drain any XML in the raw text.
                        native_tool_calls = True
                        if raw_text:
                            for etype, edata in acc.ingest_text(raw_text):
                                _emit_transformed(etype, edata, acc._base, output_lines)
                        passthrough(raw_line)
                        continue

                    if raw_text:
                        for etype, edata in acc.ingest_text(raw_text):
                            _emit_transformed(etype, edata, acc._base, output_lines)

                    if finish_reason is None:
                        if not raw_text:
                            # role-only / reasoning_content / keep-alive chunks
                            passthrough(raw_line)
                        continue

                    # ── Finish chunk handling ─────────────────────────────────
                    acc._seen_finish = True
                    if acc.has_tool_calls and not native_tool_calls:
                        # XML tool calls can be preceded by model chatter like
                        # "I'll fetch that...". Once a tool call is detected,
                        # return only tool-call deltas so clients do not display
                        # the XML-planning preamble as assistant text.
                        output_lines = [_chunk_to_line(_tool_call_delta_chunk(tc, acc._base)) for tc in acc._tool_calls]
                        output_lines.append(_chunk_to_line(finish_chunk(
                            "tool_calls",
                            parsed.get("id", ""), parsed.get("created", 0),
                        )))
                    elif finish_reason == "tool_calls" and not native_tool_calls:
                        # Model signaled tool_calls but neither llama-server nor the
                        # XML extractor produced any — emit a final chunk so the
                        # response is not empty (e.g. malformed XML).
                        _log.warning(
                            "proxy_chat  finish_reason=tool_calls but no tool_calls "
                            "captured — emitting empty final chunk",
                        )
                        output_lines.append(_chunk_to_line(finish_chunk(
                            "tool_calls",
                            parsed.get("id", ""), parsed.get("created", 0),
                        )))
                    else:
                        # Native tool_calls passed through, or plain stop — keep
                        # llama-server's original finish chunk (incl. timings).
                        passthrough(raw_line)

                if (acc.has_tool_calls or native_tool_calls) and not acc._seen_finish:
                    if acc.has_tool_calls and not native_tool_calls:
                        output_lines = [_chunk_to_line(_tool_call_delta_chunk(tc, acc._base)) for tc in acc._tool_calls]
                    output_lines.append(_chunk_to_line(finish_chunk("tool_calls")))

                if not output_lines:
                    _log.warning("proxy_chat  llama-server produced no output lines!")
                    return Response(content=b"", status_code=200, media_type="text/event-stream")

                _log.info(
                    "proxy_chat  done  output_lines=%d  tool_calls=%d  content_chars=%d",
                    len(output_lines), len(acc._tool_calls),
                    sum(len(p) for p in acc._content_parts),
                )

                async def stream():
                    for line in output_lines:
                        yield line

                return StreamingResponse(
                    stream(),
                    status_code=llama_resp.status_code,
                    media_type="text/event-stream",
                )

    except httpx.HTTPError as e:
        _log.error("llama.cpp upstream error: %s", e)
        return JSONResponse({"error": {"message": str(e), "type": "upstream_error"}}, status_code=502)


def _tool_call_delta_chunk(tool_call: dict, base: dict) -> dict:
    return {
        "id": base.get("id", "chatcmpl-glm"),
        "object": "chat.completion.chunk",
        "created": base.get("created", 0),
        "model": base.get("model", "glm-4.7-flash-llamacpp"),
        "choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}],
    }


def _emit_transformed(etype: str, edata: str | dict, base: dict, output: list[bytes]) -> None:
    """Append a transformed delta chunk to output lines."""
    if etype == "tool_call":
        output.append(_chunk_to_line(_tool_call_delta_chunk(edata, base)))
    elif etype == "content" and edata:
        output.append(_chunk_to_line({
            "id": base.get("id", "chatcmpl-glm"),
            "object": "chat.completion.chunk",
            "created": base.get("created", 0),
            "model": base.get("model", "glm-4.7-flash-llamacpp"),
            "choices": [{"index": 0, "delta": {"content": edata}, "finish_reason": None}],
        }))


async def proxy_models(request: Request) -> Response:
    """Proxy /v1/models — pass through."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{LLAMA_BASE}/v1/models", timeout=5.0)
            return Response(content=resp.content, status_code=resp.status_code,
                           headers={"content-type": "application/json"})
    except httpx.HTTPError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=502)


async def proxy_health(request: Request) -> Response:
    """Proxy /health."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{LLAMA_BASE}/health", timeout=5.0)
            return Response(content=resp.content, status_code=resp.status_code)
    except httpx.HTTPError:
        return JSONResponse({"status": "ok", "xml_proxy": "active"}, status_code=200)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN — starts XML proxy in background thread, then LiteLLM in main thread
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import litellm
    import sys

    # ── Sanitisers (disabled for 4-tool diagnostic run) ─────────────────────────
    # try:
    #     import glm_compress
    #     glm_compress.register()
    #     _log.info("glm_compress sanitizer registered")
    # except ImportError:
    #     pass
    #
    # try:
    #     import glm_token_tracker
    #     glm_token_tracker.register()
    #     _log.info("glm_token_tracker registered")
    # except ImportError:
    #     pass

    # ── Build XML proxy Starlette app ────────────────────────────────────────────
    xml_proxy_app = Starlette(
        routes=[
            Route("/v1/chat/completions", proxy_chat, methods=["POST"]),
            Route("/v1/models", proxy_models, methods=["GET"]),
            Route("/health", proxy_health, methods=["GET"]),
            Route("/", lambda r: HTMLResponse("<h1>GLM XML Tool Call Proxy</h1>"), methods=["GET"]),
        ],
        middleware=[],
    )

    # ── Patch in CORS ───────────────────────────────────────────────────────────
    xml_proxy_app = CORSMiddleware(
        xml_proxy_app,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    xml_config = uvicorn.Config(
        xml_proxy_app, host="0.0.0.0", port=PROXY_PORT, log_level="debug",
        log_config=None,  # suppress uvicorn's default JSON formatter
    )
    # Ensure our _log logger emits DEBUG and above
    _log.setLevel(logging.DEBUG)
    for h in _log.handlers:
        h.setLevel(logging.DEBUG)
    xml_server = uvicorn.Server(xml_config)

    # Start XML proxy in background thread (so we can start LiteLLM in main)
    def run_xml():
        _log.info("XML transform proxy starting on port %d → llama.cpp %d", PROXY_PORT, LLAMA_PORT)
        xml_server.run()

    xml_thread = threading.Thread(target=run_xml, daemon=True, name="xml-proxy")
    xml_thread.start()
    _log.info("XML proxy thread started (daemon)")

    # Wait for XML proxy to bind
    for _ in range(20):
        try:
            with httpx.Client(timeout=2.0) as c:
                c.get(f"http://127.0.0.1:{PROXY_PORT}/health")
            _log.info("XML proxy ready on port %d", PROXY_PORT)
            break
        except httpx.ConnectError:
            _time.sleep(0.5)
    else:
        _log.error("XML proxy failed to start on port %d", PROXY_PORT)
        sys.exit(1)

    # ── Patch LiteLLM config to point at XML proxy ──────────────────────────────
    BASE_CONFIG_PATH = ROOT / "lite_llm_config_llamacpp_direct.yaml"
    RUNTIME_CONFIG    = LOG_DIR / f"lite_llm_config_llamacpp_direct.runtime.{LLAMA_PORT}.yaml"

    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")
    # Replace direct llama-server URL with our XML proxy
    config_text = config_text.replace(
        "http://localhost:12112/v1",
        f"http://127.0.0.1:{PROXY_PORT}/v1",
    )
    RUNTIME_CONFIG.write_text(config_text, encoding="utf-8")
    _log.info("LiteLLM config patched: %s → http://127.0.0.1:%d/v1", BASE_CONFIG_PATH.name, PROXY_PORT)

    os.environ.pop("LITELLM_MASTER_KEY", None)
    os.environ.pop("LITELLM_SALT_KEY", None)
    os.environ["CONFIG_FILE_PATH"] = str(RUNTIME_CONFIG)
    os.environ["LITELLM_LOG"] = "DEBUG"
    os.environ["LITELLM_REQUEST_LOGGING"] = "true"

    # ── Start LiteLLM in main thread ────────────────────────────────────────────
    _log.info("=" * 60)
    _log.info("LiteLLM proxy starting on :%s", os.environ.get("LITE_LLM_PROXY_PORT", "12111"))
    _log.info("Full path: Copilot → LiteLLM (12111) → XML proxy (%d) → llama.cpp (%d)", PROXY_PORT, LLAMA_PORT)
    _log.info("=" * 60)

    from litellm.proxy.proxy_server import app as litellm_app

    app = CORSMiddleware(
        litellm_app,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(
        app,
        host=os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0"),
        port=int(os.environ.get("LITE_LLM_PROXY_PORT", "12111")),
        reload=False,
        log_level="info",
    )
