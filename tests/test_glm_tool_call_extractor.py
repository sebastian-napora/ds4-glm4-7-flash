import json
import sys
import types
import unittest

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:
    sys.modules["httpx"] = types.SimpleNamespace(HTTPError=Exception, AsyncClient=object)

try:
    import uvicorn  # noqa: F401
except ModuleNotFoundError:
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)

try:
    import starlette  # noqa: F401
except ModuleNotFoundError:
    starlette = types.ModuleType("starlette")
    sys.modules["starlette"] = starlette

    applications = types.ModuleType("starlette.applications")
    applications.Starlette = object
    sys.modules["starlette.applications"] = applications

    cors = types.ModuleType("starlette.middleware.cors")
    cors.CORSMiddleware = object
    sys.modules["starlette.middleware"] = types.ModuleType("starlette.middleware")
    sys.modules["starlette.middleware.cors"] = cors

    requests = types.ModuleType("starlette.requests")
    requests.Request = object
    sys.modules["starlette.requests"] = requests

    responses = types.ModuleType("starlette.responses")
    responses.HTMLResponse = object
    responses.JSONResponse = object
    responses.Response = object
    responses.StreamingResponse = object
    sys.modules["starlette.responses"] = responses

    routing = types.ModuleType("starlette.routing")
    routing.Route = object
    sys.modules["starlette.routing"] = routing

from server_compress_llamacpp_direct import (
    GLMToolCallExtractor,
    _extract_tool_names,
    _transform_non_streaming_completion,
)


class GLMToolCallExtractorTests(unittest.TestCase):
    def test_parses_normalized_tool_call_xml(self):
        extractor = GLMToolCallExtractor()

        events = extractor.ingest_text(
            "<tool_call>read_file"
            "<arg_key>file_path</arg_key><arg_value>/tmp/a.txt</arg_value>"
            "</tool_call>"
        )

        self.assertEqual(events[0][0], "tool_call")
        tool_call = events[0][1]
        self.assertEqual(tool_call["function"]["name"], "read_file")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"file_path": "/tmp/a.txt"},
        )

    def test_parses_allowed_direct_tool_xml(self):
        extractor = GLMToolCallExtractor(allowed_tool_names={"runSubagent"})

        events = extractor.ingest_text(
            "<runSubagent>"
            "<prompt>Find validation schemas</prompt>"
            "<description>Find validation schemas and error messages</description>"
            "<agentName>Explore</agentName>"
            "</runSubagent>"
        )

        self.assertEqual(events[0][0], "tool_call")
        tool_call = events[0][1]
        self.assertEqual(tool_call["function"]["name"], "runSubagent")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {
                "prompt": "Find validation schemas",
                "description": "Find validation schemas and error messages",
                "agentName": "Explore",
            },
        )

    def test_direct_tool_xml_can_arrive_split_across_chunks(self):
        extractor = GLMToolCallExtractor(allowed_tool_names={"runSubagent"})

        first_events = extractor.ingest_text("<runSub")
        second_events = extractor.ingest_text("agent><prompt>Search</prompt></runSubagent>")

        self.assertEqual(first_events, [])
        self.assertEqual(second_events[0][0], "tool_call")
        tool_call = second_events[0][1]
        self.assertEqual(tool_call["function"]["name"], "runSubagent")
        self.assertEqual(json.loads(tool_call["function"]["arguments"]), {"prompt": "Search"})

    def test_ignores_direct_xml_for_unavailable_tools(self):
        extractor = GLMToolCallExtractor(allowed_tool_names={"read_file"})

        events = extractor.ingest_text("<runSubagent><prompt>Search</prompt></runSubagent>")

        self.assertEqual(events, [("content", "<runSubagent><prompt>Search</prompt></runSubagent>")])

    def test_extract_tool_names(self):
        body = json.dumps(
            {
                "tools": [
                    {"type": "function", "function": {"name": "runSubagent"}},
                    {"type": "function", "function": {"name": "read_file"}},
                ]
            }
        )

        self.assertEqual(_extract_tool_names(body), {"runSubagent", "read_file"})

    def test_maps_slash_tool_name_to_provided_underscore_name(self):
        extractor = GLMToolCallExtractor(allowed_tool_names={"MCP_Client2_get_figma_data"})

        events = extractor.ingest_text(
            "<tool_call>MCP_Client2/get_figma_data"
            "<arg_key>fileKey</arg_key><arg_value>4TVw1VoyXO1si3l2w8pJcR</arg_value>"
            "<arg_key>nodeId</arg_key><arg_value>2073-56878</arg_value>"
            "</tool_call>"
        )

        tool_call = events[0][1]
        self.assertEqual(tool_call["function"]["name"], "MCP_Client2_get_figma_data")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"fileKey": "4TVw1VoyXO1si3l2w8pJcR", "nodeId": "2073-56878"},
        )

    def test_transforms_non_streaming_completion_with_xml_tool_call(self):
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "I'll help you create a user story."
                            "<tool_call>MCP_Client2/get_figma_data"
                            "<arg_key>fileKey</arg_key><arg_value>4TVw1VoyXO1si3l2w8pJcR</arg_value>"
                            "<arg_key>nodeId</arg_key><arg_value>2073-56878</arg_value>"
                            "</tool_call>"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        transformed = _transform_non_streaming_completion(
            json.dumps(response).encode(),
            {"MCP_Client2_get_figma_data"},
        )

        self.assertIsNotNone(transformed)
        payload = json.loads(transformed)
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "MCP_Client2_get_figma_data",
        )
        self.assertEqual(
            json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
            {"fileKey": "4TVw1VoyXO1si3l2w8pJcR", "nodeId": "2073-56878"},
        )


if __name__ == "__main__":
    unittest.main()
