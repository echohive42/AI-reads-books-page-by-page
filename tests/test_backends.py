import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

import backends
from backends import (
    ClaudeBackend,
    CodexBackend,
    GrokBackend,
    OpenAIBackend,
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


# --- the OpenAI API ---------------------------------------------------------------------------------


class _Obj:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeOpenAI:
    def __init__(self, content="", error=None):
        self.content, self.error, self.requests = content, error, []
        self.chat = _Obj(completions=_Obj(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        usage = _Obj(prompt_tokens=11, completion_tokens=4, total_tokens=15)
        return _Obj(choices=[_Obj(message=_Obj(content=self.content, refusal=None))], usage=usage)


def test_openai_page_calls_use_a_strict_schema_and_the_current_default_model():
    client = FakeOpenAI(content='{"has_content": true}')
    reply = OpenAIBackend(effort="low", client=client).ask("SYSTEM", PAGE_TEXT, PAGE_SCHEMA)

    request = client.requests[0]
    assert request["model"] == backends.API_PAGE_MODEL
    assert request["response_format"]["json_schema"] == {"name": "page_notes", "strict": True, "schema": PAGE_SCHEMA}
    assert request["reasoning_effort"] == "low"
    assert request["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": PAGE_TEXT}]
    assert reply.value == {"has_content": True}
    assert reply.usage == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}


def test_openai_summary_calls_return_markdown():
    client = FakeOpenAI(content="  ## Summary  ")
    reply = OpenAIBackend(model="some-model", client=client).ask("SYSTEM", "notes")

    assert reply.value == "## Summary"
    assert "response_format" not in client.requests[0] and "reasoning_effort" not in client.requests[0]


def test_openai_errors_are_classified():
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def response(code):
        return httpx.Response(code, request=request)

    cases = [
        (openai.RateLimitError("You exceeded your current quota", response=response(429), body=None),
         UsageLimitError),
        (openai.RateLimitError("Rate limit reached for requests", response=response(429), body=None),
         TransientError),
        (openai.AuthenticationError("Incorrect API key", response=response(401), body=None), SetupError),
        (openai.APIConnectionError(request=request), TransientError),
    ]
    for error, expected in cases:
        with pytest.raises(expected):
            OpenAIBackend(client=FakeOpenAI(error=error)).ask("SYSTEM", "prompt")


def test_openai_check_explains_the_plan_backends(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SetupError, match="--backend codex"):
        OpenAIBackend().check()
