"""How the reader talks to a model: an API key (OpenAI, Anthropic, xAI, OpenRouter),
or the Claude Code, Codex and Grok CLIs that come with a plan.

Every backend does one thing: take a system prompt and a user prompt, and return
either JSON matching a schema (a page's notes) or Markdown (a summary).

The API backends send exactly those two prompts and nothing else: no tools.

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

# Default models, checked against each provider's docs on 2026-10-29.
# OpenAI (developers.openai.com/api/docs/models): Luna is the efficient model for
# high-volume work, so it reads the pages; Sol is the balanced model for the summaries.
OPENAI_PAGE_MODEL = "gpt-6-luna"
OPENAI_SUMMARY_MODEL = "gpt-6.1-sol"
# Anthropic (platform.claude.com/docs/en/about-claude/models/overview): Opus 5.5 is the
# current default model. claude-haiku-5-5 reads for a fraction of the price.
ANTHROPIC_MODEL = "claude-opus-5-5"
# xAI (docs.x.ai/docs/models): grok-4.7 is the newest Grok 4 model, the family that
# supports structured outputs.
XAI_MODEL = "grok-4.7"
# OpenRouter: the same OpenAI models, both listed with structured outputs on
# openrouter.ai/api/v1/models?supported_parameters=structured_outputs.
OPENROUTER_PAGE_MODEL = "openai/gpt-6-luna"
OPENROUTER_SUMMARY_MODEL = "openai/gpt-6.1-sol"
# Anthropic streams every call, so a long final summary never hits a request timeout;
# this caps one answer, thinking included.
ANTHROPIC_MAX_TOKENS = 64000

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
    r"|credit balance|insufficient_quota|quota exceeded|exceeded your current quota"
    r"|insufficient credits|spending limit",
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


# Providers echo part of a rejected key in their error text ("sk-proj-****jCcA"); never print it.
_KEY_LIKE = re.compile(r"\b(?:sk|xai|key)-[A-Za-z0-9_*.\-]{6,}")


def _redact(error: Exception | str) -> str:
    return _KEY_LIKE.sub("[key hidden]", str(error))


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "…" + text[-limit:]


class Backend:
    name = "backend"
    page_model: str | None = None  # None: the CLI's own default
    summary_model: str | None = None

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


_PLAN_HINT = "or read with a plan you already have: --backend claude, --backend codex or --backend grok."


class OpenAICompatibleBackend(Backend):
    """Chat Completions through the openai SDK. OpenAI, xAI and OpenRouter all speak it."""

    provider = ""
    key_env = ""
    base_url: str | None = None
    extra_headers: dict[str, str] = {}

    def __init__(self, model: str | None = None, effort: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                 client: Any = None, base_url: str | None = None):
        super().__init__(model or self.page_model, effort, timeout)
        self.client = client
        if base_url:
            self.base_url = base_url

    def check(self) -> None:
        if self.client is None and not os.environ.get(self.key_env):
            raise SetupError(f"{self.key_env} is not set. Export your {self.provider} API key, {_PLAN_HINT}")

    def _client(self) -> Any:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.environ.get(self.key_env), base_url=self.base_url, timeout=self.timeout,
                max_retries=0,  # the reader does its own retries
                default_headers=self.extra_headers or None,
            )
        return self.client

    def request(self, system: str, prompt: str, schema: dict[str, Any] | None) -> dict[str, Any]:
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
        return request

    def model_hint(self) -> str:
        return f"{self.provider} does not know the model {self.model!r}. Pick another with --model."

    def api_error(self, exc: Exception) -> BackendError:
        import openai

        status, text = getattr(exc, "status_code", None), _redact(exc)
        if isinstance(exc, openai.AuthenticationError):
            return SetupError(f"{self.provider} rejected the key in {self.key_env}: {text}")
        if status == 402 or (status in (400, 403, 429) and _USAGE_LIMIT.search(text)):
            return UsageLimitError(f"The {self.provider} account is out of credits or quota: {text}")
        if isinstance(exc, openai.NotFoundError):
            return SetupError(f"{self.model_hint()} ({text})")
        if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)) \
                or status in (408, 409):
            return TransientError(f"{self.provider} API: {text}")
        return BackendError(f"{self.provider} API: {text}")

    def ask(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> Reply:
        import openai

        started = time.monotonic()
        try:
            response = self._client().chat.completions.create(**self.request(system, prompt, schema))
        except openai.APIError as exc:
            raise self.api_error(exc) from exc
        seconds = time.monotonic() - started
        # OpenRouter can answer 200 with an error in the body instead of choices.
        error = (getattr(response, "model_extra", None) or {}).get("error")
        if not getattr(response, "choices", None):
            message = error.get("message") if isinstance(error, dict) else error
            raise TransientError(f"{self.provider} sent no answer: {message or 'empty response'}")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "error":
            raise TransientError(f"{self.provider}: the model stopped with an error partway through")
        message = choice.message
        if getattr(message, "refusal", None):
            raise TransientError(f"The model refused: {message.refusal}")
        text = message.content or ""
        usage = getattr(response, "usage", None)
        tokens = _usage(*(getattr(usage, name, None) for name in ("prompt_tokens", "completion_tokens", "total_tokens")))
        return Reply(_parse_json(text) if schema else text.strip(), tokens, seconds)


class OpenAIBackend(OpenAICompatibleBackend):
    """The original way: the OpenAI API with OPENAI_API_KEY.

    Docs: https://developers.openai.com/api/docs/guides/structured-outputs
    """

    name = "openai"
    provider = "OpenAI"
    key_env = "OPENAI_API_KEY"
    page_model = OPENAI_PAGE_MODEL
    summary_model = OPENAI_SUMMARY_MODEL


class XAIBackend(OpenAICompatibleBackend):
    """The xAI API with XAI_API_KEY, through its OpenAI-compatible endpoint.

    Docs: https://docs.x.ai/docs/guides/structured-outputs (response_format json_schema,
    Grok 4 models) and https://docs.x.ai/docs/guides/reasoning (reasoning_effort).
    """

    name = "xai"
    provider = "xAI"
    key_env = "XAI_API_KEY"
    base_url = "https://api.x.ai/v1"
    page_model = XAI_MODEL
    summary_model = XAI_MODEL


class OpenRouterBackend(OpenAICompatibleBackend):
    """OpenRouter with OPENROUTER_API_KEY: one key for models from many providers.

    Not every model or provider supports a JSON schema, and by default OpenRouter may
    quietly drop it, so page calls set provider.require_parameters: only providers that
    honour the schema get them.
    Docs: https://openrouter.ai/docs/features/structured-outputs
    and https://openrouter.ai/docs/features/provider-routing
    """

    name = "openrouter"
    provider = "OpenRouter"
    key_env = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"
    page_model = OPENROUTER_PAGE_MODEL
    summary_model = OPENROUTER_SUMMARY_MODEL
    # Optional headers that name the app on openrouter.ai (docs: api-reference/overview).
    extra_headers = {
        "HTTP-Referer": "https://github.com/echohive42/AI-reads-books-page-by-page",
        "X-OpenRouter-Title": "AI reads books page by page",
    }

    def request(self, system: str, prompt: str, schema: dict[str, Any] | None) -> dict[str, Any]:
        request = super().request(system, prompt, schema)
        extra: dict[str, Any] = {}
        # OpenRouter's documented form is a reasoning object (docs: use-cases/reasoning-tokens).
        if request.pop("reasoning_effort", None):
            extra["reasoning"] = {"effort": self.effort}
        if schema:
            extra["provider"] = {"require_parameters": True}
        if extra:
            request["extra_body"] = extra
        return request

    def model_hint(self) -> str:
        return (
            f"OpenRouter could not serve {self.model!r}. Check the model ID; for reading pages it also has to "
            "support structured outputs: https://openrouter.ai/models?supported_parameters=structured_outputs"
        )

    def api_error(self, exc: Exception) -> BackendError:
        # 503: "no available model provider that meets your routing requirements" (docs: api-reference/errors).
        if getattr(exc, "status_code", None) == 503 and re.search(r"routing|provider", str(exc), re.IGNORECASE):
            return SetupError(f"{self.model_hint()} ({_redact(exc)})")
        return super().api_error(exc)


class AnthropicBackend(Backend):
    """The Anthropic API (Claude) with ANTHROPIC_API_KEY, through Anthropic's own SDK.

    The native SDK is used on purpose: Anthropic's OpenAI-compatible endpoint ignores
    response_format (https://platform.claude.com/docs/en/api/openai-sdk).
    Docs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
    (output_config.format), https://platform.claude.com/docs/en/build-with-claude/streaming
    and https://platform.claude.com/docs/en/api/errors
    """

    name = "anthropic"
    page_model = ANTHROPIC_MODEL
    summary_model = ANTHROPIC_MODEL

    def __init__(self, model: str | None = None, effort: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                 client: Any = None):
        super().__init__(model or self.page_model, effort, timeout)
        self.client = client

    def check(self) -> None:
        # The SDK also accepts ANTHROPIC_AUTH_TOKEN or a profile from `ant auth login`.
        if self.client is None and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                                        or shutil.which("ant")):
            raise SetupError(f"ANTHROPIC_API_KEY is not set. Export your Anthropic API key, {_PLAN_HINT}")

    def _client(self) -> Any:
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic(timeout=self.timeout, max_retries=0)  # the reader does its own retries
        return self.client

    def request(self, system: str, prompt: str, schema: dict[str, Any] | None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            request["output_config"] = output_config
        return request

    def api_error(self, exc: Exception) -> BackendError:
        import anthropic

        status, text = getattr(exc, "status_code", None), _redact(exc)
        if status == 402 or (status in (400, 403, 429) and _USAGE_LIMIT.search(text)):
            return UsageLimitError(f"The Anthropic account is out of credits: {text}")
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return SetupError(f"Anthropic rejected the key: {text}")
        if isinstance(exc, anthropic.NotFoundError):
            return SetupError(f"Anthropic does not know the model {self.model!r}. Pick another with --model. ({text})")
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError)) \
                or (isinstance(status, int) and (status >= 500 or status in (408, 409))):
            return TransientError(f"Anthropic API: {text}")  # 529 overloaded is a 5xx too
        return BackendError(f"Anthropic API: {text}")

    def ask(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> Reply:
        import anthropic

        started = time.monotonic()
        try:
            # Streamed and collected: the SDK's advice for long inputs and outputs.
            with self._client().messages.stream(**self.request(system, prompt, schema)) as stream:
                message = stream.get_final_message()
        except anthropic.APIError as exc:
            raise self.api_error(exc) from exc
        except (anthropic.CredentialsError, TypeError) as exc:
            if isinstance(exc, TypeError) and "authentication" not in str(exc):
                raise
            raise SetupError(f"No Anthropic credentials found: export ANTHROPIC_API_KEY. ({_redact(exc)})") from exc
        seconds = time.monotonic() - started
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise TransientError(f"Claude declined this request: {getattr(details, 'explanation', None) or 'refusal'}")
        if message.stop_reason == "max_tokens":
            raise TransientError(f"Claude's answer was cut off at {ANTHROPIC_MAX_TOKENS} tokens")
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        usage = message.usage
        prompt_tokens = sum(
            getattr(usage, name, None) or 0
            for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        tokens = _usage(prompt_tokens, getattr(usage, "output_tokens", None))
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


# API keys first, then the CLIs that come with a plan.
BACKENDS: dict[str, type[Backend]] = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "xai": XAIBackend,
    "openrouter": OpenRouterBackend,
    "claude": ClaudeBackend,
    "codex": CodexBackend,
    "grok": GrokBackend,
}


def make_backend(name: str, model: str | None = None, effort: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT, base_url: str | None = None) -> Backend:
    try:
        backend_class = BACKENDS[name]
    except KeyError:
        raise SetupError(f"Unknown backend {name!r}; choose one of {', '.join(BACKENDS)}") from None
    if base_url and backend_class is OpenAIBackend:
        return OpenAIBackend(model=model, effort=effort, timeout=timeout, base_url=base_url)
    return backend_class(model=model, effort=effort, timeout=timeout)
