"""The bearer key is sent, and an open port is not readiness.

No test asserted either before this file existed. `HttpPolicy` set the
`Authorization` header at `policies.py:99-100` with nothing pinning it, so the
header could have been dropped in a refactor and every test would still pass
against a local server that does not check keys -- while the deployed proxy, which
is the only exposed service, would reject every request.

Readiness has the sharper failure. The probe kept here polls `GET /v1/models`
through the proxy **with the key**; the surviving alternative
(`scripts/colab-l4-smoke.py:183-187`) polls `/health` on the raw vLLM port and so
reports ready for a server whose key check would refuse the very next call.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from smolqwen.inference.client import (
    BASE_URL_ENV,
    ChatClient,
    ReadinessError,
    normalize_base_url,
    readiness_base_url,
    wait_for_readiness,
)


class Response:
    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status = status
        self._body: Any = body if body is not None else {"choices": []}

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._body).encode()


def test_a_root_and_a_versioned_base_url_normalize_to_one_form() -> None:
    assert normalize_base_url("http://host:8000") == "http://host:8000/v1"
    assert normalize_base_url("http://host:8000/") == "http://host:8000/v1"
    assert normalize_base_url("http://host:8000/v1") == "http://host:8000/v1"
    assert normalize_base_url("http://host:8000/v1/") == "http://host:8000/v1"


def test_chat_sends_the_bearer_key() -> None:
    seen: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, *, timeout: float) -> Response:
        seen.append(request)
        assert timeout == 12.0
        return Response()

    client = ChatClient("http://host:8000", api_key="secret", timeout_s=12.0, opener=opener)
    client.chat({"model": "smolqwen", "messages": []})

    request = seen[0]
    assert request.full_url == "http://host:8000/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert request.headers["Content-type"] == "application/json"


def test_no_key_means_no_authorization_header_rather_than_an_empty_one() -> None:
    """An empty bearer is a 401 that reads like a server fault; absent is honest."""
    seen: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, *, timeout: float) -> Response:
        seen.append(request)
        return Response()

    ChatClient("http://host:8000", opener=opener).chat({"messages": []})
    assert "Authorization" not in seen[0].headers


def test_a_non_object_response_body_is_rejected() -> None:
    def opener(_request: urllib.request.Request, *, timeout: float) -> Response:
        return Response(body=[])

    client = ChatClient("http://host:8000", opener=opener)
    with pytest.raises(ValueError, match="not a JSON object"):
        client.chat({"messages": []})


def test_readiness_probes_the_authenticated_model_path() -> None:
    seen: list[Any] = []

    def opener(request: Any, *, timeout: float) -> Response:
        seen.extend([request, timeout])
        return Response()

    wait_for_readiness(
        base_url="http://proxy:8080",
        api_key="secret",
        timeout_s=1.0,
        poll_interval_s=0.1,
        opener=opener,
        monotonic=lambda: 0.0,
    )

    request = seen[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "http://proxy:8080/v1/models"
    assert request.headers["Authorization"] == "Bearer secret"


def test_a_listening_port_that_refuses_the_key_is_not_ready() -> None:
    """The failure the `/health` probe cannot see: the server is up, the key is
    rejected, and every subsequent request 401s."""
    clock = iter([0.0, 0.0, 0.5, 1.5])

    def opener(_request: Any, *, timeout: float) -> Response:
        return Response(status=401)

    with pytest.raises(ReadinessError, match="HTTP 401"):
        wait_for_readiness(
            base_url="http://proxy:8080",
            api_key="wrong",
            timeout_s=1.0,
            poll_interval_s=0.1,
            opener=opener,
            sleep=lambda _: None,
            monotonic=lambda: next(clock),
        )


def test_a_refused_connection_is_retried_until_the_deadline() -> None:
    attempts: list[int] = []
    clock = iter([0.0, 0.0, 0.4, 0.8, 1.2])

    def opener(_request: Any, *, timeout: float) -> Response:
        attempts.append(1)
        raise urllib.error.URLError("connection refused")

    with pytest.raises(ReadinessError, match="connection refused"):
        wait_for_readiness(
            base_url="http://proxy:8080",
            api_key="secret",
            timeout_s=1.0,
            poll_interval_s=0.1,
            opener=opener,
            sleep=lambda _: None,
            monotonic=lambda: next(clock),
        )
    assert len(attempts) == 3


def test_the_probe_target_prefers_an_explicit_base_url_over_the_local_proxy() -> None:
    assert readiness_base_url({}, proxy_port=8080) == "http://127.0.0.1:8080"
    assert (
        readiness_base_url({BASE_URL_ENV: "http://tunnel:443"}, proxy_port=8080)
        == "http://tunnel:443"
    )
