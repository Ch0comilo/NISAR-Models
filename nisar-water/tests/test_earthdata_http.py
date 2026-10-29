"""Offline authentication/range boundaries with synthetic credentials only."""
from io import BytesIO
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from nisar_water.earthdata import (
    AUTH_HOSTS, EarthdataError, EarthdataHTTP, TransferLedger,
    _Redirects, environment_token,
)


TOKEN = "SYNTHETIC_TEST_VALUE_NOT_AN_EARTHDATA_CREDENTIAL"
SCIENTIFIC = "https://nisar.asf.earthdatacloud.nasa.gov/test/object.h5"
CATALOG = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


class Response:
    def __init__(self, payload=b"abcdefgh", *, status=206, headers=None):
        self.payload = BytesIO(payload)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Range": "bytes 0-7/1024", "ETag": "fixture"}
        self.read_sizes = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self, size):
        assert size > 0
        self.read_sizes.append(size)
        return self.payload.read(size)


class Opener:
    def __init__(self, response=None, error=None):
        self.response = response or Response()
        self.error = error
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def client(tmp_path, *, response=None, error=None, token_provider=lambda: TOKEN):
    opener = Opener(response, error)
    ledger = TransferLedger(tmp_path / "ledger.json", limit=1024**2)
    return EarthdataHTTP(ledger, token_provider=token_provider, opener=opener), opener, ledger


@pytest.mark.parametrize("host", sorted(AUTH_HOSTS))
def test_bearer_is_sent_only_for_allowed_scientific_ranges(tmp_path, host):
    http, opener, ledger = client(tmp_path)
    assert http.read_range(f"https://{host}/fixture.h5", 0, 8) == b"abcdefgh"
    request, _ = opener.requests[0]
    assert request.get_header("Authorization") == "Bearer " + TOKEN
    assert request.get_header("Range") == "bytes=0-7"
    assert request.get_header("Accept-encoding") == "identity"
    assert ledger.total == 8
    saved = (tmp_path / "ledger.json").read_text()
    assert TOKEN not in saved and "Authorization" not in saved and "fixture.h5" not in saved
    assert json.loads(saved)["requests"][0]["host"] == host


def test_public_cmr_does_not_obtain_or_send_a_token(tmp_path):
    def forbidden_provider():
        raise AssertionError("Public metadata must not request a credential")
    response = Response(b'{"items": []}', status=200, headers={"Content-Length": "13"})
    http, opener, ledger = client(tmp_path, response=response, token_provider=forbidden_provider)
    result = http.public_json(CATALOG, {"page_size": 1, "sort_key[]": ["start_date", "producer_granule_id"]}, max_bytes=128)
    assert result == {"items": []}
    request, _ = opener.requests[0]
    assert request.get_header("Authorization") is None
    assert "page_size=1" in request.full_url and "sort_key%5B%5D=start_date" in request.full_url
    assert ledger.total == 13


@pytest.mark.parametrize("url", [
    "http://nisar.asf.earthdatacloud.nasa.gov/object.h5",
    "https://example.com/object.h5",
    "https://nisar.asf.earthdatacloud.nasa.gov.evil.example/object.h5",
    "https://user:password@nisar.asf.earthdatacloud.nasa.gov/object.h5",
    "https://nisar.asf.earthdatacloud.nasa.gov/object.h5?" + "token" + "=unsafe",
])
def test_unsupported_initial_urls_are_rejected_before_auth_or_network(tmp_path, url):
    def forbidden_provider():
        raise AssertionError("Invalid endpoint must be rejected before token access")
    http, opener, _ = client(tmp_path, token_provider=forbidden_provider)
    with pytest.raises(EarthdataError) as error:
        http.read_range(url, 0, 8)
    assert error.value.code == "UNAUTHORIZED_ENDPOINT" and not opener.requests


@pytest.mark.parametrize("url", [
    "https://cmr.earthdata.nasa.gov/signed-response",
    "https://exampledistribution.cloudfront.net/object.h5?Signature=opaque",
])
def test_redirect_removes_auth_before_public_or_cloudfront_destination(url):
    request = Request(SCIENTIFIC, headers={"Authorization": "Bearer " + TOKEN, "Range": "bytes=0-7"})
    redirected = _Redirects().redirect_request(request, None, 302, "Found", {}, url)
    assert redirected.full_url == url
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Range") == "bytes=0-7"
    # Redirect processing must not mutate the original reusable request headers.
    assert request.get_header("Authorization") == "Bearer " + TOKEN


@pytest.mark.parametrize("url", [
    "https://example.com/object.h5",
    "https://distribution.cloudfront.net.evil.example/object.h5",
    "http://exampledistribution.cloudfront.net/object.h5",
])
def test_redirect_blocks_foreign_or_non_https_destinations(url):
    request = Request(SCIENTIFIC, headers={"Authorization": "Bearer " + TOKEN})
    with pytest.raises(EarthdataError) as error:
        _Redirects().redirect_request(request, None, 302, "Found", {}, url)
    assert error.value.code == "UNAUTHORIZED_REDIRECT"
    assert TOKEN not in str(error.value)


def test_server_ignoring_range_is_closed_without_reading_full_body(tmp_path):
    response = Response(b"not a bounded scientific response", status=200,
                        headers={"Content-Length": str(20 * 1024**3)})
    http, _, ledger = client(tmp_path, response=response)
    with pytest.raises(EarthdataError) as error:
        http.read_range(SCIENTIFIC, 0, 8)
    assert error.value.code == "RANGE_NOT_SUPPORTED"
    assert response.closed and not response.read_sizes and ledger.total == 0


def test_mismatched_content_range_is_rejected_before_body_read(tmp_path):
    response = Response(headers={"Content-Range": "bytes 1-8/1024"})
    http, _, _ = client(tmp_path, response=response)
    with pytest.raises(EarthdataError) as error:
        http.read_range(SCIENTIFIC, 0, 8)
    assert error.value.code == "RANGE_NOT_SUPPORTED" and not response.read_sizes


def test_truncated_response_records_only_received_body_and_has_no_secret(tmp_path):
    http, _, ledger = client(tmp_path, response=Response(b"abc"))
    with pytest.raises(EarthdataError) as error:
        http.read_range(SCIENTIFIC, 0, 8)
    assert error.value.code == "INCOMPLETE_RANGE" and ledger.total == 3
    assert TOKEN not in str(error.value) and TOKEN not in (tmp_path / "ledger.json").read_text()


def test_budget_refuses_request_before_obtaining_token_or_network(tmp_path):
    def forbidden_provider():
        raise AssertionError("Budget must be checked before token access")
    http, opener, ledger = client(tmp_path, token_provider=forbidden_provider)
    ledger.record("cmr.earthdata.nasa.gov", 200, ledger.limit-4, "synthetic_fixture")
    with pytest.raises(EarthdataError) as error:
        http.read_range(SCIENTIFIC, 0, 8)
    assert error.value.code == "TRANSFER_LIMIT" and not opener.requests
    resumed = TransferLedger(tmp_path / "ledger.json", limit=1024**2)
    assert resumed.total == ledger.limit-4


@pytest.mark.parametrize("status,category", [(401, "AUTHENTICATION_FAILED"), (403, "AUTHORIZATION_OR_AGREEMENT_REQUIRED")])
def test_auth_failures_do_not_echo_service_body_or_credentials(tmp_path, status, category):
    body = BytesIO(("Server body echoes " + TOKEN).encode())
    error = HTTPError(SCIENTIFIC, status, "Untrusted " + TOKEN, {}, body)
    http, _, ledger = client(tmp_path, error=error)
    with pytest.raises(EarthdataError) as raised:
        http.read_range(SCIENTIFIC, 0, 8)
    assert raised.value.code == category
    assert TOKEN not in str(raised.value) and body.tell() == 0
    assert TOKEN not in (tmp_path / "ledger.json").read_text()
    assert ledger.requests[-1]["status"] == status


def test_network_failure_diagnostics_never_include_the_untrusted_reason(tmp_path):
    http, _, _ = client(tmp_path, error=URLError("untrusted " + TOKEN))
    with pytest.raises(EarthdataError) as raised:
        http.read_range(SCIENTIFIC, 0, 8)
    assert raised.value.code == "NETWORK_FAILURE" and TOKEN not in str(raised.value)


def test_environment_token_accepts_bearer_prefix_without_persisting_it(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "  Bearer " + TOKEN + "  ")
    assert environment_token() == TOKEN
    monkeypatch.setenv("EARTHDATA_TOKEN", "bad value " + TOKEN)
    with pytest.raises(EarthdataError) as raised:
        environment_token()
    assert raised.value.code == "TOKEN_REQUIRED" and TOKEN not in str(raised.value)
