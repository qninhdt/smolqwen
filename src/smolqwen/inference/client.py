"""One OpenAI-compatible HTTP surface: readiness, and chat completions.

Two things live here, and both were previously reachable only through code that
Phase 8 deletes.

`wait_for_readiness` refuses an open port as readiness: it polls
`GET /v1/models` **through the key-checking proxy, with the bearer token**, so a
vLLM process that is listening but not yet serving, and a proxy that is rejecting
the key, both read as not-ready. The surviving alternative
(`scripts/colab-l4-smoke.py:183-187`) polls `/health` on the raw vLLM port, which
bypasses the bearer check entirely -- so this is the probe worth keeping.

`ChatClient` is base-URL normalization plus the bearer header, factored out of
`eval/policies.py:76-100`. The opener stays injectable for the same reason it was
injectable there: these paths are tested without a server.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

# vLLM's own default; the environment variable overrides it where a proxy sits on
# a different port.
BASE_URL_ENV = "SMOLQWEN_BASE_URL"


class ReadinessError(RuntimeError):
    """Raised when an endpoint did not become authenticated-ready in time."""


def normalize_base_url(endpoint: str) -> str:
    """Return a `/v1`-rooted base URL, whether or not the caller supplied one."""
    trimmed = endpoint.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def wait_for_readiness(
    *,
    base_url: str,
    api_key: str,
    timeout_s: float,
    poll_interval_s: float,
    opener: Any = urllib.request.urlopen,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> None:
    """Wait for the authenticated model-list path, not merely an open port."""
    models_url = f"{normalize_base_url(base_url)}/models"
    deadline = monotonic() + timeout_s
    last_error = "endpoint did not respond"
    while monotonic() < deadline:
        request = urllib.request.Request(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with opener(request, timeout=poll_interval_s) as response:
                if int(response.status) == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        sleep(poll_interval_s)
    raise ReadinessError(
        f"authenticated model readiness did not succeed within {timeout_s:g}s: {last_error}"
    )


def readiness_base_url(environment: Mapping[str, str], *, proxy_port: int) -> str:
    """Where to probe: the environment's override, else the local proxy port."""
    return environment.get(BASE_URL_ENV, f"http://127.0.0.1:{proxy_port}")


class ChatClient:
    """A chat-completions caller that always sends the bearer key when it has one."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        opener: Any = urllib.request.urlopen,
    ) -> None:
        self.base_url = normalize_base_url(endpoint)
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._opener = opener

    @property
    def completion_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST one chat completion and return the decoded body."""
        request = urllib.request.Request(
            self.completion_url,
            data=json.dumps(dict(payload)).encode(),
            headers=self.headers(),
            method="POST",
        )
        with self._opener(request, timeout=self.timeout_s) as response:
            body: Any = json.loads(response.read().decode())
        if not isinstance(body, dict):
            raise ValueError("chat completion response is not a JSON object")
        return body
