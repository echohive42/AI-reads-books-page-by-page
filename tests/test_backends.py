import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

import backends
from backends import (
    AnthropicBackend,
    BackendError,
    ClaudeBackend,
    CodexBackend,
    GrokBackend,
    OpenAIBackend,
    OpenRouterBackend,
    XAIBackend,
    SetupError,
    TransientError,
    UsageLimitError,
)

PAGE_SCHEMA = {
    "type": "object",
    "properties": {"has_content": {"type": "boolean"}},
    "required": ["has_content"],
    "additionalProperties": False,
}
PAGE_TEXT = "SECRET PAGE TEXT that must only travel on stdin or in a prompt file"
FORBIDDEN = (
    "--dangerously-skip-permissions",
    "bypassPermissions",
    "--always-approve",
    "--yolo",
    "--dangerously-bypass-approvals-and-sandbox",
    "danger-full-access",
)


def build(backend, tmp_path, schema=PAGE_SCHEMA, system="SYSTEM"):
    work, files = tmp_path / "empty", tmp_path / "io"
    work.mkdir()
    files.mkdir()
    argv, stdin, env = backend.command(system, PAGE_TEXT, schema, work, files)
    return argv, stdin, env, work, files


def value_after(argv, flag):
    return argv[argv.index(flag) + 1]


# --- every CLI is called locked down ---------------------------------------------------------


@pytest.mark.parametrize("backend_class", [ClaudeBackend, CodexBackend, GrokBackend])
def test_no_cli_call_uses_an_always_approve_flag_or_carries_the_page_in_argv(backend_class, tmp_path):
    backend = backend_class(model="m", effort="low", executable="/bin/fake")
    argv, stdin, _env, work, _files = build(backend, tmp_path)

    joined = " ".join(argv)
    assert not any(flag in joined for flag in FORBIDDEN)
    assert PAGE_TEXT not in joined
    assert list(work.iterdir()) == []  # the model's working folder stays empty
    assert PAGE_TEXT in (stdin or (tmp_path / "io" / "prompt.txt").read_text(encoding="utf-8"))


def test_claude_runs_with_no_tools_no_mcp_no_session_and_a_schema(tmp_path):
    argv, stdin, env, _work, _files = build(ClaudeBackend(model="opus", effort="low", executable="claude"), tmp_path)

    assert argv[:2] == ["claude", "-p"]
    assert value_after(argv, "--output-format") == "json"
    assert value_after(argv, "--tools") == ""
    assert value_after(argv, "--permission-mode") == "dontAsk"
    assert {"--no-session-persistence", "--strict-mcp-config", "--safe-mode"} <= set(argv)
    assert value_after(argv, "--system-prompt") == "SYSTEM"
    assert json.loads(value_after(argv, "--json-schema")) == PAGE_SCHEMA
    assert (value_after(argv, "--model"), value_after(argv, "--effort")) == ("opus", "low")
    assert stdin == PAGE_TEXT and env is None


def test_claude_summary_calls_have_no_schema(tmp_path):
    argv, *_ = build(ClaudeBackend(executable="claude"), tmp_path, schema=None)

    assert "--json-schema" not in argv and "--model" not in argv


def test_codex_runs_read_only_ephemeral_in_an_empty_folder_with_its_tools_off(tmp_path):
    argv, stdin, _env, work, files = build(CodexBackend(model="gpt-x", effort="low", executable="codex"), tmp_path)

    assert argv[:2] == ["codex", "exec"]
    assert value_after(argv, "-s") == "read-only"
    assert {"--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules"} <= set(argv)
    assert value_after(argv, "-C") == str(work)
    assert Path(value_after(argv, "-o")).parent == files
    assert json.loads(Path(value_after(argv, "--output-schema")).read_text(encoding="utf-8")) == PAGE_SCHEMA
    disabled = {argv[i + 1] for i, item in enumerate(argv) if item == "--disable"}
    assert {"shell_tool", "unified_exec"} <= disabled
    assert 'web_search="disabled"' in argv and 'model_reasoning_effort="low"' in argv
    assert value_after(argv, "-m") == "gpt-x"
    assert argv[-1] == "-" and stdin == f"SYSTEM\n\n{PAGE_TEXT}"


def test_grok_runs_in_a_strict_sandbox_with_every_tool_removed(tmp_path):
    argv, stdin, env, work, files = build(GrokBackend(model="grok-x", effort="low", executable="grok"), tmp_path)

    assert value_after(argv, "--cwd") == str(work)
    assert value_after(argv, "--sandbox") == "strict"
    assert value_after(argv, "--permission-mode") == "dontAsk"
    assert {"--no-subagents", "--disable-web-search"} <= set(argv)
    removed = value_after(argv, "--disallowed-tools").split(",")
    assert {"run_terminal_command", "read_file", "write", "web_fetch", "Agent"} <= set(removed)
    denied = {argv[i + 1] for i, item in enumerate(argv) if item == "--deny"}
    assert {"MCPTool(*)", "Bash", "Read", "Edit", "Write"} <= denied
    assert int(value_after(argv, "--max-turns")) <= 3
    prompt_file = Path(value_after(argv, "--prompt-file"))
    assert prompt_file.is_absolute() and prompt_file.parent == files  # a relative path would resolve in --cwd
    assert json.loads(value_after(argv, "--json-schema")) == PAGE_SCHEMA
    assert stdin is None
    assert env["GROK_MEMORY"] == "0" and env["GROK_CLAUDE_MCPS_ENABLED"] == "0"


# --- reading each CLI's answer ---------------------------------------------------------------------


def claude_json(**fields):
    return json.dumps({"type": "result", "is_error": False, "result": "", **fields})


def test_claude_reads_structured_output_and_counts_tokens(tmp_path):
    backend = ClaudeBackend(executable="claude")
    stdout = claude_json(
        structured_output={"has_content": True},
        usage={"input_tokens": 5, "cache_read_input_tokens": 100, "output_tokens": 7},
    )

    value, usage = backend.parse(0, stdout, "", PAGE_SCHEMA, tmp_path)

    assert value == {"has_content": True}
    assert usage == {"prompt_tokens": 105, "completion_tokens": 7, "total_tokens": 112}


def test_claude_falls_back_to_json_in_the_result_text(tmp_path):
    stdout = claude_json(result='```json\n{"has_content": false}\n```')

    value, _usage = ClaudeBackend(executable="claude").parse(0, stdout, "", PAGE_SCHEMA, tmp_path)

    assert value == {"has_content": False}


def test_claude_markdown_answer_and_errors(tmp_path):
    backend = ClaudeBackend(executable="claude")
    assert backend.parse(0, claude_json(result="  ## Summary\n"), "", None, tmp_path)[0] == "## Summary"
    with pytest.raises(SetupError, match="not logged in"):
        backend.parse(1, claude_json(is_error=True, result="Not logged in · Please run /login"), "", None, tmp_path)
    with pytest.raises(UsageLimitError):
        backend.parse(1, claude_json(is_error=True, result="You've hit your usage limit"), "", None, tmp_path)
    with pytest.raises(TransientError):
        backend.parse(0, claude_json(result="not json"), "", PAGE_SCHEMA, tmp_path)


def test_codex_reads_the_answer_file_and_its_token_count(tmp_path):
    (tmp_path / "answer.txt").write_text('{"has_content": true}', encoding="utf-8")

    value, usage = CodexBackend(executable="codex").parse(0, "", "tokens used\n1,234\n", PAGE_SCHEMA, tmp_path)

    assert value == {"has_content": True} and usage == {"total_tokens": 1234}


def test_codex_errors_are_classified(tmp_path):
    backend = CodexBackend(executable="codex")
    with pytest.raises(UsageLimitError):
        backend.parse(1, "", "noise\nERROR: You've hit your usage limit. Try again later.\n", None, tmp_path)
    with pytest.raises(SetupError, match="not logged in"):
        backend.parse(1, "", "Error: not logged in. Run codex login\n", None, tmp_path)


def test_grok_reads_structured_output_or_text(tmp_path):
    backend = GrokBackend(executable="grok")
    usage = {"input_tokens": 3, "output_tokens": 4}
    page = json.dumps({"text": "", "structuredOutput": {"has_content": True}, "usage": usage})
    summary = json.dumps({"text": " ## Notes ", "usage": usage})

    assert backend.parse(0, page, "", PAGE_SCHEMA, tmp_path) == ({"has_content": True},
                                                                 {"prompt_tokens": 3, "completion_tokens": 4,
                                                                  "total_tokens": 7})
    assert backend.parse(0, summary, "", None, tmp_path)[0] == "## Notes"
    with pytest.raises(SetupError):
        backend.parse(1, json.dumps({"error": "not authenticated: run grok login"}), "", None, tmp_path)
    with pytest.raises(TransientError):
        backend.parse(0, json.dumps({"text": ""}), "", None, tmp_path)


@pytest.mark.parametrize("message, error, hint", [
    ("You've hit your usage limit. Upgrade to Pro", UsageLimitError, "usage limit"),
    ("ERROR: exceeded your current quota", UsageLimitError, "usage limit"),
    ("Invalid API key · Please run /login", SetupError, "not logged in"),
    ("The model 'gpt-nope' does not exist or you do not have access", SetupError, "not available"),
    ("error: unexpected argument '--output-schema' found", SetupError, "too old"),
    ("stream disconnected before completion", TransientError, "stream"),
])
def test_cli_errors_become_the_right_kind_with_a_hint(message, error, hint):
    exc = CodexBackend(model="gpt-nope", executable="codex").fail(message)

    assert type(exc) is error and hint in str(exc)


def test_a_missing_cli_is_a_clear_setup_error(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda _name: None)

    for backend in (ClaudeBackend(), CodexBackend(), GrokBackend()):
        with pytest.raises(SetupError, match="was not found"):
            backend.check()


def test_unknown_backend_name():
    with pytest.raises(SetupError, match="Unknown backend"):
        backends.make_backend("gemini")


# --- real subprocesses with a stand-in CLI -----------------------------------------------------------

FAKE_CLI = r'''
import json, os, sys, time
argv = sys.argv[1:]
mode = os.environ.get("FAKE_CLI_MODE", "")
stdin = sys.stdin.read()
log = {"argv": argv, "cwd": os.getcwd(), "cwd_files": os.listdir("."), "stdin": stdin,
       "grok_memory": os.environ.get("GROK_MEMORY")}
if mode == "sleep":
    child = os.fork()
    if child == 0:
        time.sleep(60)
        os._exit(0)
    log["child"] = child
with open(os.environ["FAKE_CLI_LOG"], "w") as handle:
    json.dump(log, handle)
if mode == "sleep":
    time.sleep(60)
page = {"has_content": True, "page_gist": "g", "knowledge": ["k"]}
schema = "--json-schema" in argv or "--output-schema" in argv
if argv[0] == "exec":
    answer = argv[argv.index("-o") + 1]
    with open(answer, "w") as handle:
        handle.write(json.dumps(page) if schema else "## codex notes")
    sys.stderr.write("tokens used\n42\n")
elif "--prompt-file" in argv:
    prompt = open(argv[argv.index("--prompt-file") + 1]).read()
    log["prompt_file"] = prompt
    with open(os.environ["FAKE_CLI_LOG"], "w") as handle:
        json.dump(log, handle)
    print(json.dumps({"text": "" if schema else "## grok notes", "structuredOutput": page if schema else None,
                      "usage": {"input_tokens": 1, "output_tokens": 2}}))
elif mode == "limit":
    print(json.dumps({"type": "result", "is_error": True, "result": "You've hit your usage limit"}))
    sys.exit(1)
else:
    print(json.dumps({"type": "result", "is_error": False, "result": "## claude notes",
                      "structured_output": page if schema else None, "usage": {"output_tokens": 2}}))
'''


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("the stand-in CLI is a POSIX script")
    script = tmp_path / "bin" / "fake-cli"
    script.parent.mkdir()
    interpreter = sys.executable if " " not in sys.executable else "/usr/bin/env python3"
    script.write_text(f"#!{interpreter}\n{FAKE_CLI}", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "log.json"
    monkeypatch.setenv("FAKE_CLI_LOG", str(log))
    monkeypatch.delenv("FAKE_CLI_MODE", raising=False)
    return str(script), lambda: json.loads(log.read_text(encoding="utf-8"))


@pytest.mark.parametrize("backend_class, answer", [
    (ClaudeBackend, "## claude notes"),
    (CodexBackend, "## codex notes"),
    (GrokBackend, "## grok notes"),
])
def test_each_cli_gets_the_page_privately_in_an_empty_folder(backend_class, answer, fake_cli):
    executable, read_log = fake_cli
    backend = backend_class(executable=executable, timeout=30)

    reply = backend.ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    log = read_log()
    assert reply.value == {"has_content": True, "page_gist": "g", "knowledge": ["k"]}
    assert reply.seconds > 0
    assert log["cwd_files"] == [] and Path(log["cwd"]).name == "empty"
    assert PAGE_TEXT not in " ".join(log["argv"])
    assert PAGE_TEXT in (log["stdin"] or log.get("prompt_file", ""))
    assert backend.ask("SYSTEM", "summarise", None).value == answer
    assert not Path(log["cwd"]).exists()  # the temporary folders are cleaned up
    if backend_class is GrokBackend:
        assert log["grok_memory"] == "0"


def test_a_plan_limit_from_a_real_subprocess_stops_the_reader(fake_cli, monkeypatch):
    executable, _read_log = fake_cli
    monkeypatch.setenv("FAKE_CLI_MODE", "limit")

    with pytest.raises(UsageLimitError):
        ClaudeBackend(executable=executable, timeout=30).ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)


def test_a_cli_that_hangs_is_stopped_with_everything_it_started(fake_cli, monkeypatch):
    executable, read_log = fake_cli
    monkeypatch.setenv("FAKE_CLI_MODE", "sleep")
    started = time.monotonic()

    with pytest.raises(TransientError, match="did not answer"):
        ClaudeBackend(executable=executable, timeout=2).ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    assert time.monotonic() - started < 15
    child = read_log()["child"]
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the CLI's child process was left running")


# --- the API backends: OpenAI, xAI, OpenRouter (openai SDK) and Anthropic (its own SDK) -----------


class _Obj:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeOpenAI:
    def __init__(self, content="", error=None, response=None):
        self.content, self.error, self.response, self.requests = content, error, response, []
        self.chat = _Obj(completions=_Obj(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        if self.response is not None:
            return self.response
        usage = _Obj(prompt_tokens=11, completion_tokens=4, total_tokens=15)
        choice = _Obj(message=_Obj(content=self.content, refusal=None), finish_reason="stop")
        return _Obj(choices=[choice], usage=usage)


def openai_error(kind, message, code):
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    if kind == "connection":
        return openai.APIConnectionError(request=request)
    return getattr(openai, kind)(message, response=httpx.Response(code, request=request), body=None)


def test_openai_page_calls_use_a_strict_schema_and_the_current_default_model():
    client = FakeOpenAI(content='{"has_content": true}')
    reply = OpenAIBackend(effort="low", client=client).ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    request = client.requests[0]
    assert request["model"] == backends.OPENAI_PAGE_MODEL
    assert request["response_format"]["json_schema"] == {"name": "page_notes", "strict": True, "schema": PAGE_SCHEMA}
    assert request["reasoning_effort"] == "low"
    assert request["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": PAGE_TEXT}]
    assert "tools" not in request and "extra_body" not in request
    assert reply.value == {"has_content": True}
    assert reply.usage == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}


def test_openai_summary_calls_return_markdown():
    client = FakeOpenAI(content="  ## Summary  ")
    reply = OpenAIBackend(model="some-model", client=client).ask("SYSTEM", "notes")

    assert reply.value == "## Summary"
    assert "response_format" not in client.requests[0] and "reasoning_effort" not in client.requests[0]


def test_each_openai_compatible_provider_has_its_own_endpoint_key_and_default_model(monkeypatch):
    for name in ("OPENAI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.setenv(name, f"test-{name}") if name != "OPENAI_BASE_URL" else monkeypatch.delenv(name, raising=False)

    openai_client = OpenAIBackend()._client()
    xai_client = XAIBackend()._client()
    router_client = OpenRouterBackend()._client()
    custom_client = OpenAIBackend(base_url="https://compatible.example.test/v1")._client()

    assert str(openai_client.base_url).startswith("https://api.openai.com/v1")
    assert str(xai_client.base_url).startswith("https://api.x.ai/v1")
    assert str(router_client.base_url).startswith("https://openrouter.ai/api/v1")
    assert str(custom_client.base_url).startswith("https://compatible.example.test/v1")
    assert openai_client.api_key == "test-OPENAI_API_KEY" and xai_client.api_key == "test-XAI_API_KEY"
    assert router_client.api_key == "test-OPENROUTER_API_KEY"
    assert router_client.max_retries == 0 and xai_client.max_retries == 0  # the reader retries, with backoff
    assert router_client.default_headers["X-OpenRouter-Title"] == "AI reads books page by page"
    assert (XAIBackend().model, OpenRouterBackend().model) == (backends.XAI_MODEL, backends.OPENROUTER_PAGE_MODEL)


def test_xai_uses_the_openai_shape_with_reasoning_effort():
    client = FakeOpenAI(content='{"has_content": false}')
    XAIBackend(effort="high", client=client).ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    request = client.requests[0]
    assert request["model"] == "grok-4.7" and request["reasoning_effort"] == "high"
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True


def test_openrouter_only_sends_page_calls_to_providers_that_honour_the_schema():
    client = FakeOpenAI(content='{"has_content": true}')
    OpenRouterBackend(effort="medium", client=client).ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)
    OpenRouterBackend(client=client).ask("SYSTEM", "notes")

    page, summary = client.requests
    assert page["extra_body"] == {"reasoning": {"effort": "medium"}, "provider": {"require_parameters": True}}
    assert "reasoning_effort" not in page and page["response_format"]["json_schema"]["schema"] == PAGE_SCHEMA
    assert "extra_body" not in summary and "response_format" not in summary


def test_openrouter_errors_inside_a_200_answer_are_retried():
    no_choices = _Obj(choices=None, model_extra={"error": {"code": 502, "message": "upstream went away"}})
    broke_off = _Obj(choices=[_Obj(message=_Obj(content='{"has', refusal=None), finish_reason="error")],
                     model_extra={})

    with pytest.raises(TransientError, match="upstream went away"):
        OpenRouterBackend(client=FakeOpenAI(response=no_choices)).ask("SYSTEM", "prompt", PAGE_SCHEMA)
    with pytest.raises(TransientError, match="partway"):
        OpenRouterBackend(client=FakeOpenAI(response=broke_off)).ask("SYSTEM", "prompt", PAGE_SCHEMA)


@pytest.mark.parametrize("backend_class", [OpenAIBackend, XAIBackend, OpenRouterBackend])
@pytest.mark.parametrize("kind, message, code, expected", [
    ("RateLimitError", "You exceeded your current quota", 429, UsageLimitError),
    ("RateLimitError", "Rate limit reached for requests", 429, TransientError),
    ("AuthenticationError", "Incorrect API key", 401, SetupError),
    ("APIStatusError", "Insufficient credits. Add more credits and retry the request.", 402, UsageLimitError),
    ("PermissionDeniedError", "Your team has run out of credits or hit its spending limit", 403, UsageLimitError),
    ("PermissionDeniedError", "Flagged by moderation", 403, BackendError),
    ("NotFoundError", "The model does not exist", 404, SetupError),
    ("APIStatusError", "Request timed out", 408, TransientError),
    ("InternalServerError", "Bad gateway", 502, TransientError),
    ("BadRequestError", "Invalid schema", 400, BackendError),
    ("connection", "", 0, TransientError),
])
def test_api_errors_are_classified(backend_class, kind, message, code, expected):
    with pytest.raises(expected) as caught:
        backend_class(client=FakeOpenAI(error=openai_error(kind, message, code))).ask("SYSTEM", "prompt")

    if expected is BackendError:
        assert type(caught.value) is BackendError  # not retried, not a setup problem


def test_openrouter_explains_a_model_that_cannot_return_a_schema():
    error = openai_error("InternalServerError",
                         "There is no available model provider that meets your routing requirements", 503)

    with pytest.raises(SetupError, match="supported_parameters=structured_outputs"):
        OpenRouterBackend(client=FakeOpenAI(error=error)).ask("SYSTEM", "prompt", PAGE_SCHEMA)


@pytest.mark.parametrize("backend_class, key", [
    (OpenAIBackend, "OPENAI_API_KEY"), (XAIBackend, "XAI_API_KEY"), (OpenRouterBackend, "OPENROUTER_API_KEY"),
])
def test_a_missing_api_key_explains_the_plan_backends(backend_class, key, monkeypatch):
    monkeypatch.delenv(key, raising=False)

    with pytest.raises(SetupError, match=f"{key} is not set.*--backend codex"):
        backend_class().check()


class FakeAnthropic:
    def __init__(self, content=None, stop_reason="end_turn", error=None, stop_details=None):
        self.content = content if content is not None else []
        self.stop_reason, self.error, self.stop_details, self.requests = stop_reason, error, stop_details, []
        self.messages = _Obj(stream=self.stream)

    def stream(self, **request):
        self.requests.append(request)
        fake = self

        class Stream:
            def __enter__(self):
                if fake.error:
                    raise fake.error
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                usage = _Obj(input_tokens=20, output_tokens=7, cache_read_input_tokens=5,
                             cache_creation_input_tokens=None)
                return _Obj(content=fake.content, stop_reason=fake.stop_reason, stop_details=fake.stop_details,
                            usage=usage)

        return Stream()


def text_block(text):
    return _Obj(type="text", text=text)


def test_anthropic_page_calls_use_output_config_with_the_schema_and_no_tools():
    thinking = _Obj(type="thinking", thinking="")
    client = FakeAnthropic(content=[thinking, text_block('{"has_content": true, "knowledge": ["a"]}')])
    reply = AnthropicBackend(effort="low", client=client).ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == "SYSTEM"
    assert request["messages"] == [{"role": "user", "content": PAGE_TEXT}]
    assert request["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": PAGE_SCHEMA}}
    assert request["max_tokens"] == backends.ANTHROPIC_MAX_TOKENS
    for unsupported in ("tools", "tool_choice", "temperature", "top_p", "top_k", "thinking"):
        assert unsupported not in request
    assert reply.value == {"has_content": True, "knowledge": ["a"]}
    assert reply.usage == {"prompt_tokens": 25, "completion_tokens": 7, "total_tokens": 32}


def test_anthropic_summary_calls_return_markdown_without_output_config():
    client = FakeAnthropic(content=[text_block("  ## Summary  ")])
    reply = AnthropicBackend(model="claude-haiku-5-5", client=client).ask("SYSTEM", "notes")

    assert reply.value == "## Summary"
    assert client.requests[0]["model"] == "claude-haiku-5-5" and "output_config" not in client.requests[0]


def test_anthropic_refusals_and_cut_off_answers_are_not_parsed():
    details = _Obj(category=None, explanation="not this one")

    with pytest.raises(TransientError, match="not this one"):
        AnthropicBackend(client=FakeAnthropic(content=[text_block("{")], stop_reason="refusal",
                                              stop_details=details)).ask("SYSTEM", "p", PAGE_SCHEMA)
    with pytest.raises(TransientError, match="cut off"):
        AnthropicBackend(client=FakeAnthropic(content=[text_block('{"has')], stop_reason="max_tokens")).ask(
            "SYSTEM", "p", PAGE_SCHEMA)


def anthropic_error(kind, message, code):
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    if kind == "connection":
        return anthropic.APIConnectionError(request=request)
    return getattr(anthropic, kind)(message, response=httpx2.Response(code, request=request), body=None)


@pytest.mark.parametrize("kind, message, code, expected", [
    ("AuthenticationError", "invalid x-api-key", 401, SetupError),
    ("PermissionDeniedError", "Your API key does not have permission", 403, SetupError),
    ("PermissionDeniedError", "Your credit balance is too low", 403, UsageLimitError),
    ("BadRequestError", "Your credit balance is too low to access the Anthropic API", 400, UsageLimitError),
    ("APIStatusError", "billing_error", 402, UsageLimitError),
    ("NotFoundError", "model: claude-nope", 404, SetupError),
    ("RateLimitError", "Number of request tokens has exceeded your per-minute rate limit", 429, TransientError),
    ("InternalServerError", "Internal server error", 500, TransientError),
    ("OverloadedError", "Overloaded", 529, TransientError),
    ("BadRequestError", "output_config.format.schema: unsupported", 400, BackendError),
    ("connection", "", 0, TransientError),
])
def test_anthropic_errors_are_classified(kind, message, code, expected):
    with pytest.raises(expected) as caught:
        AnthropicBackend(client=FakeAnthropic(error=anthropic_error(kind, message, code))).ask("SYSTEM", "p")

    if expected is BackendError:
        assert type(caught.value) is BackendError


def test_anthropic_without_credentials_is_a_clear_setup_error(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(backends.shutil, "which", lambda _name: None)

    with pytest.raises(SetupError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicBackend().check()


# --- the real SDKs, offline: what each provider would receive on the wire -----------------------------


class Wire:
    """A stand-in HTTP server for the real openai and anthropic SDKs."""

    def __init__(self, status=200, body=None, sse=None):
        self.status, self.body, self.sse, self.requests = status, body, sse, []

    def handler(self, httpx_module):
        def handle(request):
            self.requests.append(request)
            if self.sse is not None:
                return httpx_module.Response(self.status, headers={"content-type": "text/event-stream"},
                                             content=self.sse.encode())
            return httpx_module.Response(self.status, json=self.body)

        return handle

    def sent(self):
        request = self.requests[-1]
        return str(request.url), request.headers, json.loads(request.content)


def chat_answer(content):
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content, "refusal": None}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }


@pytest.fixture
def openai_wire(monkeypatch):
    import httpx
    import openai

    wire = Wire(body=chat_answer('{"has_content": true}'))
    real = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(wire.handler(httpx)))))
    for name in ("OPENAI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return wire


def test_openrouter_on_the_wire(openai_wire):
    reply = OpenRouterBackend(effort="low").ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    url, headers, body = openai_wire.sent()
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-OPENROUTER_API_KEY"
    assert headers["x-openrouter-title"] == "AI reads books page by page"
    assert body["model"] == backends.OPENROUTER_PAGE_MODEL
    assert body["provider"] == {"require_parameters": True} and body["reasoning"] == {"effort": "low"}
    assert body["response_format"]["json_schema"]["strict"] is True and "reasoning_effort" not in body
    assert reply.value == {"has_content": True} and reply.usage["total_tokens"] == 12


def test_openrouter_error_body_with_status_200_on_the_wire(openai_wire):
    openai_wire.body = {"error": {"code": 502, "message": "Provider returned error"}}

    with pytest.raises(TransientError, match="Provider returned error"):
        OpenRouterBackend().ask("SYSTEM", "prompt", PAGE_SCHEMA)


def test_xai_and_openai_on_the_wire(openai_wire):
    XAIBackend(effort="high").ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)
    url, headers, body = openai_wire.sent()
    assert url == "https://api.x.ai/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-XAI_API_KEY"
    assert (body["model"], body["reasoning_effort"]) == ("grok-4.7", "high")
    assert body["response_format"]["type"] == "json_schema"

    OpenAIBackend().ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)
    url, headers, body = openai_wire.sent()
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-OPENAI_API_KEY"
    assert body["model"] == backends.OPENAI_PAGE_MODEL and "x-openrouter-title" not in headers


def test_a_402_from_the_wire_stops_the_reader(openai_wire):
    openai_wire.status, openai_wire.body = 402, {"error": {"code": 402, "message": "Insufficient credits"}}

    with pytest.raises(UsageLimitError):
        OpenRouterBackend().ask("SYSTEM", "prompt")


def anthropic_sse(text, stop_reason="end_turn"):
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 30, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "signature_delta", "signature": "sig"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 9}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


@pytest.fixture
def anthropic_wire(monkeypatch):
    import anthropic
    import httpx2

    wire = Wire(sse=anthropic_sse('{"has_content": true, "page_gist": "g", "knowledge": ["k"]}'))
    real = anthropic.Anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: real(
        **kw, http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(wire.handler(httpx2)))))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    return wire


def test_anthropic_on_the_wire(anthropic_wire):
    reply = AnthropicBackend(effort="medium").ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    url, headers, body = anthropic_wire.sent()
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "test-anthropic-key"
    assert body["stream"] is True and body["model"] == "claude-opus-5-5" and body["system"] == "SYSTEM"
    assert body["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": PAGE_SCHEMA}}
    assert body["messages"] == [{"role": "user", "content": PAGE_TEXT}]
    assert reply.value == {"has_content": True, "page_gist": "g", "knowledge": ["k"]}
    assert reply.usage == {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39}


def test_anthropic_overloaded_on_the_wire_is_retried(anthropic_wire):
    anthropic_wire.sse = None
    anthropic_wire.status = 529
    anthropic_wire.body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

    with pytest.raises(TransientError, match="Overloaded"):
        AnthropicBackend().ask("SYSTEM", "prompt")


def test_a_rejected_key_is_never_echoed_back():
    message = ("Incorrect API key provided: sk-proj-****************************jCcA. "
               "You can find your API key at https://platform.openai.com/account/api-keys.")
    error = openai_error("AuthenticationError", message, 401)

    with pytest.raises(SetupError) as caught:
        OpenAIBackend(client=FakeOpenAI(error=error)).ask("SYSTEM", "prompt")

    assert "jCcA" not in str(caught.value) and "sk-proj" not in str(caught.value)
    assert "[key hidden]" in str(caught.value) and "platform.openai.com/account/api-keys" in str(caught.value)
    assert backends._redact("xai-AbCdEf123456 and sk-ant-api03-abcDEF_123") == "[key hidden] and [key hidden]"
