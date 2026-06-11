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

from server_compress_llamacpp_direct import GLMToolCallExtractor, _extract_tool_names


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


if __name__ == "__main__":
    unittest.main()
