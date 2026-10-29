"""In-memory Bearer authentication and bounded NASA/ASF HTTPS requests.

No credential persistence, credential URLs or full-granule fallback. Each client
belongs to one worker job; transfer ledgers contain hosts/counters, not tokens.
"""
from pathlib import Path
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .common import write_json

AUTH_HOSTS = frozenset({"nisar.asf.earthdatacloud.nasa.gov", "datapool.asf.alaska.edu",
                        "datapool.asf.earthdatacloud.nasa.gov", "urs.earthdata.nasa.gov"})
PUBLIC_HOSTS = frozenset({"cmr.earthdata.nasa.gov"})


class EarthdataError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


def environment_token():
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token or any(c.isspace() for c in token):
        raise EarthdataError("TOKEN_REQUIRED", "Supply your Earthdata Bearer token in memory or EARTHDATA_TOKEN.")
    return token


def _allowed(url, *, redirect=False):
    parsed = urlsplit(url)
    return (parsed.scheme == "https" and parsed.port in (None, 443)
            and not parsed.username and not parsed.password and not parsed.fragment
            and (redirect or not parsed.query)
            and (parsed.hostname in AUTH_HOSTS | PUBLIC_HOSTS
                 or (redirect and (parsed.hostname or "").endswith(".cloudfront.net"))))


class _Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _allowed(newurl, redirect=True):
            raise EarthdataError("UNAUTHORIZED_REDIRECT", "The data service redirected outside supported HTTPS hosts.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(newurl).hostname not in AUTH_HOSTS:
            redirected.remove_header("Authorization")
        return redirected


class TransferLedger:
    def __init__(self, path, *, limit=128 * 1024**2, storage_root=None):
        self.path = Path(path)
        self.storage_root = Path(storage_root or self.path.parent)
        self.limit = int(limit)
        if not 1024**2 <= self.limit <= 6 * 1024**3:
            raise ValueError("Transfer limit must be 1 MiB to 6 GiB")
        self.total = 0
        self.requests = []
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.total = int(saved["body_bytes"])
            self.requests = saved["requests"]
        if self.total > self.limit:
            raise EarthdataError("TRANSFER_LIMIT", "The existing job already exceeds its transfer limit.")

    def check(self, maximum):
        if not 0 < maximum <= 8 * 1024**2 or self.total + maximum > self.limit:
            raise EarthdataError("TRANSFER_LIMIT", "The bounded transfer budget would be exceeded.")

    def record(self, host, status, count, category):
        self.total += int(count)
        self.requests.append({"host": host, "status": status, "bytes": int(count), "category": category})
        write_json(self.path, {"schema": "nisar-water-transfer-1", "body_bytes": self.total,
                              "limit_bytes": self.limit, "requests": self.requests})


class EarthdataHTTP:
    def __init__(self, ledger, *, token_provider=environment_token, timeout=30, opener=None):
        self.ledger = ledger
        self.token_provider = token_provider
        self.timeout = min(60, max(1, float(timeout)))
        self.opener = opener or build_opener(_Redirects())
        self._sizes = {}

    def _request(self, url, *, start=None, end=None, metadata_limit=2 * 1024**2, parameters=None):
        if not _allowed(url):
            raise EarthdataError("UNAUTHORIZED_ENDPOINT", "Only unsigned supported NASA/ASF HTTPS source URLs are accepted.")
        host = urlsplit(url).hostname
        ranged = start is not None
        if ranged and (host not in AUTH_HOSTS or start < 0 or end <= start):
            raise EarthdataError("INVALID_RANGE", "Use an ordered bounded range on a supported scientific host.")
        if not ranged and host not in PUBLIC_HOSTS:
            raise EarthdataError("UNBOUNDED_SCIENTIFIC_READ", "Scientific objects require exact byte ranges.")
        maximum = end - start if ranged else metadata_limit
        self.ledger.check(maximum)
        headers = {"Accept-Encoding": "identity", "User-Agent": "nisar-water bounded scientific reader"}
        if ranged:
            token = self.token_provider().strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
            if not token or any(c.isspace() for c in token):
                raise EarthdataError("TOKEN_REQUIRED", "Supply a valid Earthdata Bearer token.")
            headers.update(Authorization="Bearer " + token, Range=f"bytes={start}-{end-1}")
        request_url = url + ("?" + urlencode(parameters, doseq=True) if parameters else "")
        actual, status = 0, None
        try:
            with self.opener.open(Request(request_url, headers=headers), timeout=self.timeout) as response:
                status = response.status
                if ranged:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if status != 206 or not match or int(match[1]) != start or int(match[2]) + 1 != end:
                        raise EarthdataError("RANGE_NOT_SUPPORTED", "The server did not honor the exact range; no full body was read.")
                    self._sizes[url] = {"size": int(match[3]), "etag": response.headers.get("ETag")}
                elif status != 200 or int(response.headers.get("Content-Length") or 0) > maximum:
                    raise EarthdataError("METADATA_RESPONSE", "Catalog response exceeds its bound or was unsuccessful.")
                parts = []
                while actual < maximum:
                    part = response.read(min(65536, maximum - actual))
                    if not part:
                        break
                    parts.append(part)
                    actual += len(part)
                if ranged and actual != maximum:
                    raise EarthdataError("INCOMPLETE_RANGE", "The scientific range response was truncated.")
                if not ranged and actual == maximum:
                    raise EarthdataError("METADATA_LIMIT", "Catalog response reached its size bound.")
                result = b"".join(parts)
        except HTTPError as error:
            status = error.code
            # Error bodies can contain arbitrary service content; never persist them.
            category = {401: "AUTHENTICATION_FAILED", 403: "AUTHORIZATION_OR_AGREEMENT_REQUIRED",
                        404: "GRANULE_UNAVAILABLE"}.get(status, "HTTP_SERVICE_ERROR")
            raise EarthdataError(category, f"Data service returned HTTP {status}; check your token/access and retry.") from None
        except (URLError, OSError, TimeoutError):
            raise EarthdataError("NETWORK_FAILURE", "The bounded data request failed; retry the job.") from None
        finally:
            self.ledger.record(host, status, actual, "scientific_range" if ranged else "public_catalog")
        return result

    def stat(self, url):
        if url not in self._sizes:
            self.read_range(url, 0, 8)
        return dict(self._sizes[url])

    def read_range(self, url, start, end):
        return self._request(url, start=int(start), end=int(end))

    def public_json(self, url, params=None, *, max_bytes=8 * 1024**2):
        return json.loads(self._request(url, metadata_limit=max_bytes, parameters=params))
