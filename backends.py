"""How the reader talks to a model: the OpenAI API, or the Claude Code, Codex and Grok CLIs.

Every backend does one thing: take a system prompt and a user prompt, and return
either JSON matching a schema (a page's notes) or Markdown (a summary).

The CLI backends make one locked-down, one-shot call per request. They run without a
shell, in an empty temporary folder, with every tool switched off, so the model only
ever sees the text the reader sends it: one page plus its notes so far, never the book
file. They never use an always-approve or bypass-permissions flag.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT = 600  # seconds per call; a long final summary can take a few minutes

# The OpenAI API's current models (checked against developers.openai.com/api/docs/models
# on 2026-10-29): Luna is the efficient model OpenAI names as gpt-4o-mini's replacement,
# Sol is the balanced model for the summaries.
API_PAGE_MODEL = "gpt-6-luna"
API_SUMMARY_MODEL = "gpt-6.1-sol"

# Codex features that give the model a way to run code or open files. With only
# `-s read-only` and an empty working folder, Codex 0.163 could still read any file on
# the disk through its exec tool, so these are switched off too.
CODEX_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "code_mode_host",
    "multi_agent",
    "view_image",
    "apps",
    "plugins",
    "image_generation",
)

# Grok's built-in tools (1.0.70). An unknown name in an allowlist would leave every tool
# on, so instead every built-in is removed by name, and deny rules back that up.
GROK_BUILTIN_TOOLS = (
    "Agent", "run_terminal_command", "read_file", "search_replace", "list_dir", "grep",
    "kill_command_or_subagent", "todo_write", "get_command_or_subagent_output", "monitor",
    "scheduler_create", "scheduler_delete", "scheduler_list", "search_tool", "use_tool",
    "workflow", "enter_plan_mode", "exit_plan_mode", "ask_user_question", "send_feedback",
    "image_gen", "image_edit", "image_to_video", "reference_to_video", "write",
    "web_search", "web_fetch", "open_page", "open_page_with_find",
)
# Keep the user's memory, MCP servers, hooks, skills and rules (including ones Grok
# borrows from Claude, Codex and Cursor setups) out of the reading.
GROK_QUIET_ENV = {
    "GROK_MEMORY": "0",
    "GROK_BACKEND_SEARCH": "0",
    "GROK_DISABLE_AUTOUPDATER": "1",
    "GROK_MANAGED_MCPS_ENABLED": "0",
    "GROK_MANAGED_MCP_GATEWAY_TOOLS_ENABLED": "0",
    "NO_COLOR": "1",
    **{
        f"GROK_{source}_{kind}_ENABLED": "0"
        for source in ("CLAUDE", "CODEX", "CURSOR")
        for kind in ("AGENTS", "HOOKS", "MCPS", "RULES", "SKILLS")
    },
}

# A plan or account that is out of quota (stop and resume later), not a short rate limit (retry).
_USAGE_LIMIT = re.compile(
    r"usage limit|hit your .{0,40}limit|(plan|session|daily|weekly|monthly) limit|out of credits"
    r"|credit balance|insufficient_quota|quota exceeded|exceeded your current quota",
    re.IGNORECASE,
)
_NOT_LOGGED_IN = re.compile(
    r"not logged in|not authenticated|\blog ?in\b|\bsign ?in\b|authenticat|oauth|unauthori[sz]ed|\b401\b"
    r"|invalid api key",
    re.IGNORECASE,
)
_UNKNOWN_MODEL = re.compile(
    r"model\b.{0,80}\b(not found|not supported|does not exist|unknown|not available|invalid)"
    r"|(unknown|invalid|unsupported) model",
    re.IGNORECASE,
)
_TOO_OLD = re.compile(r"unknown (option|argument|feature)|unexpected argument|unrecognized (option|argument)",
                      re.IGNORECASE)


class BackendError(RuntimeError):
    """A call failed in a way that retrying will not fix."""


class TransientError(BackendError):
    """A call failed in a way that may work on a retry (timeouts, network, bad JSON)."""


class SetupError(BackendError):
    """The backend is not ready: the CLI is missing, too old, or not logged in."""


class UsageLimitError(BackendError):
    """The plan or account hit its usage limit. Progress is saved; rerun later to resume."""


@dataclass
class Reply:
    value: Any  # a dict for schema calls, Markdown text otherwise
    usage: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def _usage(prompt: Any = None, completion: Any = None, total: Any = None) -> dict[str, int]:
    usage = {
        name: value
        for name, value in (("prompt_tokens", prompt), ("completion_tokens", completion), ("total_tokens", total))
        if isinstance(value, int)
    }
    if "total_tokens" not in usage and {"prompt_tokens", "completion_tokens"} <= usage.keys():
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "…" + text[-limit:]


class Backend:
    name = "backend"

    def __init__(self, model: str | None = None, effort: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.model = model
        self.effort = effort
        self.timeout = timeout

    def label(self) -> str:
        return f"{self.name} ({self.model or 'its default model'})"

    def check(self) -> None:
        """Fail early, before any page is read, when the backend cannot work at all."""

    def ask(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> Reply:
        raise NotImplementedError


class OpenAIBackend(Backend):
    """The original way: the OpenAI API with OPENAI_API_KEY."""

    name = "api"

    def __init__(self, model: str | None = None, effort: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                 client: Any = None):
        super().__init__(model or API_PAGE_MODEL, effort, timeout)
        self.client = client

    def check(self) -> None:
        if self.client is None and not os.environ.get("OPENAI_API_KEY"):
            raise SetupError(
                "OPENAI_API_KEY is not set. Export your key, or read with a plan you already have: "
                "--backend claude, --backend codex or --backend grok."
            )

    def _client(self) -> Any:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(timeout=self.timeout, max_retries=0)  # the reader does its own retries
        return self.client

    def ask(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> Reply:
        import openai

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if self.effort:
            request["reasoning_effort"] = self.effort
        if schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "page_notes", "strict": True, "schema": schema},
            }
        started = time.monotonic()
        try:
            response = self._client().chat.completions.create(**request)
        except openai.AuthenticationError as exc:
            raise SetupError(f"The OpenAI API rejected the key: {exc}") from exc
        except openai.RateLimitError as exc:
            if _USAGE_LIMIT.search(str(exc)):
                raise UsageLimitError(f"The OpenAI account is out of quota: {exc}") from exc
            raise TransientError(f"OpenAI rate limit: {exc}") from exc
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
            raise TransientError(f"OpenAI API: {exc}") from exc
        except openai.NotFoundError as exc:
            raise SetupError(f"OpenAI does not know the model {self.model!r}: {exc}") from exc
        seconds = time.monotonic() - started
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise TransientError(f"The model refused: {message.refusal}")
        text = message.content or ""
        usage = getattr(response, "usage", None)
        tokens = _usage(*(getattr(usage, name, None) for name in ("prompt_tokens", "completion_tokens", "total_tokens")))
        return Reply(_parse_json(text) if schema else text.strip(), tokens, seconds)


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TransientError(f"The model's answer was not valid JSON: {_tail(text, 200)}") from exc
    if not isinstance(value, dict):
        raise TransientError("The model's answer was JSON but not an object")
    return value


class CLIBackend(Backend):
    """Shared plumbing for the agent CLIs: one subprocess per call, no shell, no tools."""

    binary = ""
    install_hint = ""
    login_hint = ""

    def __init__(self, model: str | None = None, effort: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                 executable: str | None = None):
        super().__init__(model, effort, timeout)
        self.executable = executable

    def check(self) -> None:
        self._executable()

    def _executable(self) -> str:
        if self.executable is None:
            found = shutil.which(self.binary)
            if not found:
                raise SetupError(f"The `{self.binary}` command was not found. {self.install_hint}")
            self.executable = found
        return self.executable

    def ask(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> Reply:
        with tempfile.TemporaryDirectory(prefix="read-books-") as scratch:
            # The model's working folder stays empty. Files the CLI itself reads or writes
            # (schema, prompt, answer) live next to it, never inside it.
            work = Path(scratch) / "empty"
            files = Path(scratch) / "io"
            work.mkdir()
            files.mkdir()
            argv, stdin, env = self.command(system, prompt, schema, work, files)
            started = time.monotonic()
            returncode, stdout, stderr = run_cli(argv, stdin, work, env, self.timeout, self.name)
            seconds = time.monotonic() - started
            value, usage = self.parse(returncode, stdout, stderr, schema, files)
        return Reply(value, usage, seconds)

    def command(self, system: str, prompt: str, schema: dict[str, Any] | None, work: Path,
                files: Path) -> tuple[list[str], str | None, dict[str, str] | None]:
        raise NotImplementedError

    def parse(self, returncode: int, stdout: str, stderr: str, schema: dict[str, Any] | None,
              files: Path) -> tuple[Any, dict[str, int]]:
        raise NotImplementedError

    def fail(self, message: str) -> BackendError:
        """Turn a CLI's error text into the right kind of error, with a hint for the person."""
        if _USAGE_LIMIT.search(message):
            return UsageLimitError(f"{self.name}: your plan's usage limit was reached. {_tail(message, 300)}")
        if _UNKNOWN_MODEL.search(message):
            return SetupError(f"{self.name}: the model {self.model!r} is not available here. "
                              f"Try another --model, or leave it out to use the CLI's default. ({_tail(message, 300)})")
        if _TOO_OLD.search(message):
            return SetupError(f"{self.name}: this `{self.binary}` looks too old for these flags. "
                              f"Update it and try again. ({_tail(message, 300)})")
        if _NOT_LOGGED_IN.search(message):
            return SetupError(f"{self.name}: not logged in. {self.login_hint} ({_tail(message, 300)})")
        return TransientError(f"{self.name}: {_tail(message)}")


def run_cli(argv: list[str], stdin: str | None, cwd: Path, env: dict[str, str] | None, timeout: float,
            name: str) -> tuple[int, str, str]:
    """Run one CLI call without a shell; on timeout, stop it and everything it started."""
    posix = os.name != "nt"
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=posix,
        )
    except FileNotFoundError as exc:
        raise SetupError(f"Could not start {argv[0]}") from exc
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop(process, posix)
        raise TransientError(f"{name} did not answer within {timeout:.0f} s")
    except BaseException:
        _stop(process, posix)  # Ctrl+C: never leave a CLI running in the background
        raise
    return process.returncode, stdout, stderr


def _stop(process: subprocess.Popen, posix: bool) -> None:
    if process.poll() is not None:
        return
    try:
        if posix:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
        try:
            if posix:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        process.communicate(timeout=5)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


class ClaudeBackend(CLIBackend):
    """Claude Code on your Claude plan (`claude -p`), with every tool off."""

    name = "claude"
    binary = "claude"
    install_hint = "Install Claude Code: https://code.claude.com/docs"
    login_hint = "Run `claude` once and sign in with your Claude account (or `claude auth login`), then rerun."

    def command(self, system, prompt, schema, work, files):
        argv = [
            self._executable(), "-p",
            "--output-format", "json",
            "--tools", "",                 # no tools at all: it cannot open, run or fetch anything
            "--permission-mode", "dontAsk",
            "--no-session-persistence",
            "--strict-mcp-config",         # no MCP servers
            "--safe-mode",                 # no CLAUDE.md, hooks, skills or plugins in the reading
            "--system-prompt", system,
        ]
        if schema:
            argv += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        if self.model:
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--effort", self.effort]
        return argv, prompt, None  # the page goes in on stdin, never on the command line

    def parse(self, returncode, stdout, stderr, schema, files):
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            raise self.fail(stderr or stdout or f"exit code {returncode}") from None
        if not isinstance(data, dict):
            raise self.fail(stdout)
        if data.get("is_error") or returncode != 0:
            raise self.fail(str(data.get("result") or stderr or data.get("subtype") or f"exit code {returncode}"))
        usage = data.get("usage") or {}
        prompt_tokens = sum(
            usage.get(name) or 0
            for name in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        tokens = _usage(prompt_tokens or None, usage.get("output_tokens"))
        if schema:
            structured = data.get("structured_output")
            if isinstance(structured, dict):
                return structured, tokens
            return _parse_json(str(data.get("result") or "")), tokens
        return str(data.get("result") or "").strip(), tokens


class CodexBackend(CLIBackend):
    """Codex on your ChatGPT plan (`codex exec`), read-only and with every tool off."""

    name = "codex"
    binary = "codex"
    install_hint = "Install the Codex CLI from https://developers.openai.com/codex"
    login_hint = "Run `codex login` and sign in with your ChatGPT account, then rerun."

    def command(self, system, prompt, schema, work, files):
        argv = [
            self._executable(), "exec",
            "-s", "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",        # no MCP servers, profiles or hooks from ~/.codex/config.toml
            "--ignore-rules",
            "--color", "never",
            "-C", str(work),
            "-o", str(files / "answer.txt"),
            "-c", 'web_search="disabled"',
        ]
        for feature in CODEX_DISABLED_FEATURES:
            argv += ["--disable", feature]
        if schema:
            schema_path = files / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            argv += ["--output-schema", str(schema_path)]
        if self.model:
            argv += ["-m", self.model]
        if self.effort:
            argv += ["-c", f'model_reasoning_effort="{self.effort}"']
        argv.append("-")  # read the prompt from stdin
        return argv, f"{system}\n\n{prompt}", None

    def parse(self, returncode, stdout, stderr, schema, files):
        answer = files / "answer.txt"
        text = answer.read_text(encoding="utf-8") if answer.exists() else ""
        if returncode != 0 or not text.strip():
            raise self.fail(_codex_error(stderr) or stdout or f"exit code {returncode}")
        match = re.search(r"tokens used\s*\n?\s*([\d,]+)", stderr)
        tokens = _usage(total=int(match.group(1).replace(",", ""))) if match else {}
        return (_parse_json(text) if schema else text.strip()), tokens


def _codex_error(stderr: str) -> str:
    lines = [line for line in (stderr or "").splitlines() if re.search(r"error|limit|login|unknown", line, re.I)]
    return "\n".join(lines[-5:]) or stderr


class GrokBackend(CLIBackend):
    """Grok Build on your Grok plan (`grok --prompt-file`), strict sandbox and every tool removed."""

    name = "grok"
    binary = "grok"
    install_hint = "Install Grok Build, the Grok CLI: https://docs.x.ai/build/overview"
    login_hint = "Run `grok login` and sign in with your Grok account, then rerun."

    def command(self, system, prompt, schema, work, files):
        prompt_file = files / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        argv = [
            self._executable(),
            "--cwd", str(work),
            "--sandbox", "strict",
            "--permission-mode", "dontAsk",
            "--no-subagents",
            "--disable-web-search",
            "--disallowed-tools", ",".join(GROK_BUILTIN_TOOLS),
            "--deny", "MCPTool(*)",
            "--deny", "Bash",
            "--deny", "Read",
            "--deny", "Edit",
            "--deny", "Write",
            "--deny", "Grep",
            "--max-turns", "3",
            "--system-prompt-override", system,
            "--output-format", "json",
            "--prompt-file", str(prompt_file),
        ]
        if schema:
            argv += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        if self.model:
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--reasoning-effort", self.effort]
        return argv, None, {**os.environ, **GROK_QUIET_ENV}

    def parse(self, returncode, stdout, stderr, schema, files):
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            raise self.fail(stderr or stdout or f"exit code {returncode}") from None
        if returncode != 0 or not isinstance(data, dict) or data.get("error"):
            detail = data.get("error") if isinstance(data, dict) else None
            raise self.fail(str(detail or stderr or stdout or f"exit code {returncode}"))
        usage = data.get("usage") or {}
        tokens = _usage(usage.get("input_tokens"), usage.get("output_tokens"))
        if schema:
            structured = data.get("structuredOutput")
            if isinstance(structured, dict):
                return structured, tokens
            return _parse_json(str(data.get("text") or "")), tokens
        text = str(data.get("text") or "").strip()
        if not text:
            raise self.fail(stderr or "empty answer")
        return text, tokens


BACKENDS: dict[str, type[Backend]] = {
    "api": OpenAIBackend,
    "claude": ClaudeBackend,
    "codex": CodexBackend,
    "grok": GrokBackend,
}


def make_backend(name: str, model: str | None = None, effort: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> Backend:
    try:
        backend_class = BACKENDS[name]
    except KeyError:
        raise SetupError(f"Unknown backend {name!r}; choose one of {', '.join(BACKENDS)}") from None
    return backend_class(model=model, effort=effort, timeout=timeout)
