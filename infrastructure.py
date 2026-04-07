from __future__ import annotations

"""Infrastructure adapters and helpers.

This module provides:
- HTTP-based ingestion + document text adapters
- Run context creation + observability helpers
- Safe/strict JSON persistence helpers used by persistence.py

The goal is to keep the application layer (application.py) isolated from vendor/API
details while keeping runtime behavior deterministic where required.
"""

import asyncio
import contextvars
import json
import hashlib
import logging
import os
import re
import tempfile
import gzip
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from application import RunContext
from config import RuntimeConfig
from domain import RegulatoryDocumentHandle
from ports import (
    DocumentTextPort,
    RegulatoryIngestionPort,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Observability (deterministic IDs; never used in trading decisions)
# =============================================================================

_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="trace-000000")
_TRACE_COUNTER: int = 0

def obs_new_trace_id() -> str:
    """Create and set a new trace id in a deterministic (non-random) way."""
    global _TRACE_COUNTER
    _TRACE_COUNTER += 1
    trace_id = f"trace-{_TRACE_COUNTER:06d}"
    _TRACE_ID.set(trace_id)
    return trace_id

def obs_get_trace_id() -> str:
    return _TRACE_ID.get()

# Backwards-compatible helper (not used by current llm.py; kept for imports)
_DECISION_COUNTER: int = 0

def obs_new_decision_id() -> str:
    global _DECISION_COUNTER
    _DECISION_COUNTER += 1
    return f"decision-{_DECISION_COUNTER:06d}"

# =============================================================================
# Time / JSON helpers
# =============================================================================

def now_utc_iso() -> str:
    dt = _parse_env_utc_datetime(os.environ.get("RUN_NOW_UTC"))
    return (dt or datetime.now(timezone.utc)).isoformat()

def safe_json_load(path: Path, default: Any) -> Any:
    """Best-effort JSON loader. Returns default on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def strict_json_load(path: Path) -> Any:
    """Strict JSON loader. Raises on IO/parse errors."""
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)

def safe_json_save(path: Path, obj: Any) -> None:
    """Atomic JSON save (write temp then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Optional[Path] = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# =============================================================================
# Log / JSONL rotation helpers (observability only)
# =============================================================================

def rotate_jsonl_gz_if_needed(path: Path, *, max_mb: int, backup_count: int) -> None:
    """Rotate a JSONL file if it exceeds max_mb.

    Rotation format:
      <name>.<UTC timestamp>.gz

    This is observability-only and must never affect trading decisions.
    """
    try:
        p = Path(path)
        if not p.exists():
            return
        try:
            max_bytes = max(1, int(max_mb)) * 1024 * 1024
        except Exception:
            max_bytes = 50 * 1024 * 1024

        if p.stat().st_size <= max_bytes:
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        gz_path = p.with_name(f"{p.name}.{ts}.gz")

        # Compress deterministically (no metadata).
        with open(p, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6, mtime=0) as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        # Truncate original (start fresh).
        try:
            p.unlink()
        except Exception:
            # If unlink fails, just truncate.
            try:
                with open(p, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass

        # Enforce backup retention (keep newest).
        try:
            backups = sorted(p.parent.glob(p.name + ".*.gz"), key=lambda x: x.name, reverse=True)
            keep = max(0, int(backup_count))
            if keep and len(backups) > keep:
                for old in backups[keep:]:
                    try:
                        old.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        return
def strip_html_to_text(html: str) -> str:
    # Basic HTML tag stripping; deterministic and dependency-free.
    text = re.sub(r"<\s*br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n\s+\n", "\n\n", text)
    return text.strip()

# =============================================================================
# Run context
# =============================================================================

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

def _parse_env_utc_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = value.strip()
    # Accept a strict Zulu ISO format: 2026-02-15T12:34:56Z
    if _ISO_UTC_RE.match(s):
        try:
            # fromisoformat doesn't parse 'Z' in all versions; replace.
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None
    # Accept YYYYMMDDTHHMMSSZ
    if re.fullmatch(r"\d{8}T\d{6}Z", s):
        try:
            return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def create_run_context(config: RuntimeConfig) -> RunContext:
    """Create a RunContext.

    Determinism hooks:
    - RUN_ID: explicit run id (string). If absent, uses now_utc formatted.
    - RUN_NOW_UTC or RUN_AT_UTC: freeze the run clock (UTC).
      Examples: 2026-02-15T12:00:00Z or 20260215T120000Z
    """
    now_utc = _parse_env_utc_datetime(os.environ.get("RUN_NOW_UTC") or os.environ.get("RUN_AT_UTC")) or datetime.now(timezone.utc)

    run_id_raw = (os.environ.get("RUN_ID") or now_utc.strftime("%Y%m%dT%H%M%SZ")).strip()
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id_raw)

    run_dir = Path(config.runs_dir) / run_id
    console_dir = run_dir / "console"
    tables_dir = run_dir / "tables"
    artifacts_dir = run_dir / "artifacts"

    for p in (run_dir, console_dir, tables_dir, artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=run_id,
        now_utc=now_utc,
        run_dir=run_dir,
        console_dir=console_dir,
        tables_dir=tables_dir,
        artifacts_dir=artifacts_dir,
    )


def rotate_run_observability_files(ctx: RunContext, config: RuntimeConfig) -> None:
    """Rotate per-run JSONL observability files (best-effort)."""
    try:
        rotate_jsonl_gz_if_needed(Path(ctx.run_dir) / "events.jsonl", max_mb=int(getattr(config, "eventlog_max_mb", 50) or 50), backup_count=int(getattr(config, "eventlog_backup_count", 10) or 10))
        rotate_jsonl_gz_if_needed(Path(ctx.run_dir) / "stage_events.jsonl", max_mb=int(getattr(config, "eventlog_max_mb", 50) or 50), backup_count=int(getattr(config, "eventlog_backup_count", 10) or 10))
        rotate_jsonl_gz_if_needed(Path(ctx.run_dir) / "metrics.jsonl", max_mb=int(getattr(config, "eventlog_max_mb", 50) or 50), backup_count=int(getattr(config, "eventlog_backup_count", 10) or 10))
        rotate_jsonl_gz_if_needed(Path(ctx.run_dir) / "llm_calls.jsonl", max_mb=int(getattr(config, "eventlog_max_mb", 50) or 50), backup_count=int(getattr(config, "eventlog_backup_count", 10) or 10))
    except Exception:
        return


def prune_old_run_folders(config: RuntimeConfig, *, keep_run_id: str = "") -> None:
    """Best-effort pruning of old run folders to control disk usage.

    - Uses folder mtime (filesystem) as the signal.
    - Never touches runs/_shared.
    - Never touches the active keep_run_id folder.
    """
    try:
        days = int(getattr(config, "runs_prune_days", 0) or 0)
    except Exception:
        days = 0
    if days <= 0:
        return

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
    except Exception:
        return

    runs_dir = Path(getattr(config, "runs_dir"))
    if not runs_dir.exists():
        return

    for child in runs_dir.iterdir():
        try:
            if not child.is_dir():
                continue
            name = child.name
            if name == "_shared":
                continue
            if keep_run_id and name == str(keep_run_id):
                continue
            # Skip session folders that are clearly active by name? We'll rely on mtime.
            mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
            if mtime >= cutoff:
                continue
            # Delete folder tree
            for root, dirs, files in os.walk(child, topdown=False):
                for fn in files:
                    try:
                        Path(root, fn).unlink()
                    except Exception:
                        pass
                for dn in dirs:
                    try:
                        Path(root, dn).rmdir()
                    except Exception:
                        pass
            try:
                child.rmdir()
            except Exception:
                pass
        except Exception:
            continue

# =============================================================================
# HTTP ingestion + document text adapters
# =============================================================================

def _parse_http_date(s: str) -> Optional[datetime]:
    """Best-effort parse for RFC822 / RFC3339 / ISO dates.

    Returns timezone-aware UTC datetimes where possible.
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None

    # RFC822 / RFC1123 (common in RSS feeds)
    try:
        import email.utils

        dt = email.utils.parsedate_to_datetime(s)
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Atom / ISO-8601
    try:
        iso = s
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Common explicit formats
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None

async def _async_safe_json_load(path: Path, default: Any) -> Any:
    return await asyncio.to_thread(safe_json_load, path, default)

async def _async_safe_json_save(path: Path, obj: Any) -> None:
    await asyncio.to_thread(safe_json_save, path, obj)

def _http_cache_key(url: str, params: Optional[Dict[str, Any]] = None, *, drop_keys: Optional[set[str]] = None) -> str:
    """Build a deterministic cache key for conditional HTTP headers.

    We explicitly drop secret-bearing query keys (API keys) from the cache key so we do not persist them.
    """
    u = (url or "").strip()
    p = dict(params or {})
    if drop_keys:
        for k in list(p.keys()):
            if k in drop_keys:
                p.pop(k, None)

    if not p:
        return u

    try:
        from urllib.parse import urlencode

        qs = urlencode(sorted((str(k), str(v)) for k, v in p.items()), doseq=True)
        return f"{u}?{qs}"
    except Exception:
        return u

class HttpConditionalHeadersCache:
    """Very small persistent cache of ETag / Last-Modified per request key."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._loaded = False
        self._data: Dict[str, Dict[str, str]] = {}

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            data = await _async_safe_json_load(self._path, {})
            if isinstance(data, dict):
                self._data = {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}
            else:
                self._data = {}
            self._loaded = True

    async def conditional_headers(self, key: str) -> Dict[str, str]:
        await self._ensure_loaded()
        entry = self._data.get(str(key)) or {}
        if not isinstance(entry, dict):
            entry = {}
        headers: Dict[str, str] = {}
        etag = entry.get("etag")
        last_modified = entry.get("last_modified")
        if isinstance(etag, str) and etag.strip():
            headers["If-None-Match"] = etag.strip()
        if isinstance(last_modified, str) and last_modified.strip():
            headers["If-Modified-Since"] = last_modified.strip()
        return headers

    async def update_from_response(self, key: str, resp: httpx.Response) -> None:
        etag = (resp.headers.get("ETag") or "").strip()
        last_modified = (resp.headers.get("Last-Modified") or "").strip()
        if not etag and not last_modified:
            return

        await self._ensure_loaded()
        async with self._lock:
            entry = dict(self._data.get(str(key)) or {})
            if etag:
                entry["etag"] = etag
            if last_modified:
                entry["last_modified"] = last_modified
            entry["updated_at"] = now_utc_iso()
            self._data[str(key)] = entry
            await _async_safe_json_save(self._path, self._data)

def _is_retryable_status(code: int) -> bool:
    return code in {408, 425, 429, 500, 502, 503, 504}

def _deterministic_backoff_seconds(attempt_idx: int) -> float:
    # Deterministic schedule (no jitter).
    schedule = (0.5, 1.0, 2.0, 4.0)
    if attempt_idx < 0:
        return schedule[0]
    if attempt_idx >= len(schedule):
        return schedule[-1]
    return schedule[attempt_idx]

class AsyncRateLimiter:
    """Deterministic async rate limiter.

    Enforces a minimum interval between allowed requests. This is used
    to comply with SEC EDGAR fair-access expectations (moderate request
    rates; identify yourself via User-Agent).
    """

    def __init__(self, *, min_interval_seconds: float):
        self._min_interval = max(0.0, float(min_interval_seconds))
        self._lock = asyncio.Lock()
        self._next_allowed_at: float = 0.0

    async def wait(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_allowed_at > now:
                await asyncio.sleep(self._next_allowed_at - now)
                now = loop.time()
            self._next_allowed_at = now + self._min_interval


def _looks_like_sec_block_page(body_text: str) -> bool:
    """Detect SEC WAF/rate-limit HTML pages returned as 403."""
    s = (body_text or '').lower()
    return (
        'request rate threshold exceeded' in s
        or 'undeclared automated tool' in s
        or 'unclassified bot' in s
        or 'to allow for equitable access' in s
        or 'please declare your traffic' in s
    )

class AsyncHttpRequester:
    """HTTP helper that provides:
    - bounded deterministic retry w/ backoff
    - conditional GET (ETag / If-Modified-Since) via a tiny cache
    - optional host-based rate limiting

    Notes for SEC/EDGAR:
    - The SEC enforces "fair access" and may return 403 HTML pages when it
      believes traffic is coming from an undeclared automated tool or when
      request rates are too high.
    - Always send a descriptive User-Agent with contact info, and keep the
      request rate modest.
    """

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        cache: HttpConditionalHeadersCache,
        default_headers: Optional[Dict[str, str]] = None,
        max_attempts: int = 4,
        rate_limiter: AsyncRateLimiter | None = None,
        rate_limit_host_suffixes: Tuple[str, ...] = (),
        sec_block_cooldown_seconds: float | None = None,
    ):
        self._http = http
        self._cache = cache
        self._default_headers = dict(default_headers or {})
        self._max_attempts = max(1, int(max_attempts))

        self._rate_limiter = rate_limiter
        self._rate_limit_host_suffixes = tuple(rate_limit_host_suffixes or ())

        try:
            self._sec_block_cooldown_seconds = float(sec_block_cooldown_seconds) if sec_block_cooldown_seconds is not None else 0.0
        except Exception:
            self._sec_block_cooldown_seconds = 0.0

    def _host_needs_rate_limit(self, host: str) -> bool:
        if self._rate_limiter is None:
            return False
        h = (host or '').strip().lower()
        if not h:
            return False
        for suf in self._rate_limit_host_suffixes:
            s = (suf or '').strip().lower().lstrip('.')
            if s and h.endswith(s):
                return True
        return False

    async def get(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        follow_redirects: bool = False,
        use_conditional: bool = False,
        cache_key: Optional[str] = None,
    ) -> Optional[httpx.Response]:
        url = (url or '').strip()
        if not url:
            return None

        # Host extraction for optional rate limiting.
        host = ''
        try:
            from urllib.parse import urlparse

            host = (urlparse(url).hostname or '').strip().lower()
        except Exception:
            host = ''

        req_headers = dict(self._default_headers)
        if headers:
            req_headers.update({str(k): str(v) for k, v in headers.items() if k and v is not None})

        key = cache_key or _http_cache_key(url, params)
        if use_conditional:
            try:
                req_headers.update(await self._cache.conditional_headers(key))
            except Exception:
                pass

        last_err: Optional[Exception] = None

        for attempt in range(self._max_attempts):
            forced_sleep_s: float = 0.0
            try:
                if self._host_needs_rate_limit(host):
                    await self._rate_limiter.wait()  # type: ignore[union-attr]

                resp = await self._http.get(
                    url,
                    headers=req_headers,
                    params=params,
                    timeout=timeout,
                    follow_redirects=follow_redirects,
                )

                if resp.status_code == 304:
                    return None

                # SEC sometimes responds with a 403 HTML page when it thinks traffic is
                # "undeclared automation" or request rates are too high.
                if int(resp.status_code) == 403 and host.endswith('sec.gov'):
                    body = ''
                    try:
                        body = resp.text or ''
                    except Exception:
                        body = ''

                    if _looks_like_sec_block_page(body):
                        ua = (req_headers.get('User-Agent') or '').strip()
                        if '@' not in ua and 'mailto:' not in ua.lower():
                            raise RuntimeError(
                                'SEC returned HTTP 403 and appears to be blocking scripted traffic. '
                                'Set SEC_USER_AGENT to a descriptive value with contact info (email), '
                                "e.g. 'MyAppName/1.0 (you@yourdomain.com)'. "
                                'Then reduce request rate (SEC guidance: <=10 req/s; go lower).'
                            )

                        # Treat as retryable, but cool down for a while.
                        forced_sleep_s = max(forced_sleep_s, float(self._sec_block_cooldown_seconds or 600.0))
                        last_err = httpx.HTTPStatusError(
                            f'SEC 403 (rate limited / blocked) for {url}',
                            request=resp.request,
                            response=resp,
                        )
                    else:
                        resp.raise_for_status()

                elif _is_retryable_status(int(resp.status_code)):
                    last_err = httpx.HTTPStatusError(
                        f'Retryable HTTP {resp.status_code} for {url}',
                        request=resp.request,
                        response=resp,
                    )
                else:
                    resp.raise_for_status()
                    if use_conditional:
                        try:
                            await self._cache.update_from_response(key, resp)
                        except Exception:
                            pass
                    return resp

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                last_err = e
            except httpx.RequestError as e:
                last_err = e

            if attempt < (self._max_attempts - 1):
                sleep_s = _deterministic_backoff_seconds(attempt)

                # Respect Retry-After deterministically if present (429/503).
                try:
                    if isinstance(last_err, httpx.HTTPStatusError) and last_err.response is not None:
                        ra = (last_err.response.headers.get('Retry-After') or '').strip()
                        if ra:
                            ra_s = float(ra)
                            if ra_s > 0:
                                sleep_s = max(sleep_s, min(ra_s, 30.0))
                except Exception:
                    pass

                # SEC blocks may require a much longer cool-down.
                if forced_sleep_s and forced_sleep_s > 0:
                    sleep_s = max(sleep_s, min(float(forced_sleep_s), 3600.0))

                logging.info(
                    '[HTTP] retry %s/%s %s',
                    attempt + 1,
                    self._max_attempts - 1,
                    url,
                    extra={'extra_details': {'sleep_s': sleep_s}},
                )
                await asyncio.sleep(sleep_s)

        if last_err:
            raise last_err
        return None


class HttpRegulatoryIngestionAdapter(RegulatoryIngestionPort):
    """Ingestion adapter for home-exchange feeds.

    Document ingestion is handled by feed adapters in feeds.py, coordinated
    by the runner. This adapter provides the RegulatoryIngestionPort interface
    expected by the application layer use case.

    The runner must call set_feed_items() after Phase 1 feed collection so that
    ingest_documents() returns the collected items as RegulatoryDocumentHandle
    objects for Phase 2 processing.
    """

    def __init__(self, *, http: httpx.AsyncClient, config: RuntimeConfig):
        self._http = http
        self._config = config
        self._feed_items: List[Any] = []

    def set_feed_items(self, feed_items: List[Any]) -> None:
        """Populate with FeedItem objects collected during Phase 1."""
        self._feed_items = list(feed_items or [])

    async def ingest_documents(self) -> List[RegulatoryDocumentHandle]:
        """Convert Phase 1 FeedItems into RegulatoryDocumentHandle objects."""
        docs: List[RegulatoryDocumentHandle] = []
        for item in self._feed_items:
            try:
                docs.append(RegulatoryDocumentHandle(
                    doc_id=str(getattr(item, "item_id", "") or ""),
                    source=str(getattr(item, "feed", "") or ""),
                    title=str(getattr(item, "title", "") or ""),
                    published_at=getattr(item, "published_at", None),
                    url=str(getattr(item, "url", "") or ""),
                    metadata={
                        "ticker": str(getattr(item, "us_ticker", "") or "").upper().strip(),
                        "company_name": str(getattr(item, "company_name", "") or ""),
                        "home_ticker": str(getattr(item, "home_ticker", "") or ""),
                        "content_snippet": str(getattr(item, "content_snippet", "") or ""),
                        **(getattr(item, "metadata", {}) or {}),
                    },
                ))
            except Exception as exc:
                logger.warning("Failed to convert feed item to document: %s", exc)
        return docs


def _cap_text(s: str, max_chars: int = 2_000_000) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars]

def _is_probably_pdf(url: str, content_type: str) -> bool:
    u = (url or "").lower()
    ct = (content_type or "").lower()
    return u.endswith(".pdf") or "application/pdf" in ct

def _is_probably_zip(url: str, content_type: str) -> bool:
    u = (url or "").lower()
    ct = (content_type or "").lower()
    return u.endswith(".zip") or "application/zip" in ct or "application/octet-stream" in ct

def _bytes_to_text_best_effort(b: bytes) -> str:
    if not b:
        return ""
    # Try utf-8, then latin-1 as a deterministic fallback.
    try:
        return b.decode("utf-8")
    except Exception:
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            try:
                return b.decode("latin-1", errors="ignore")
            except Exception:
                return ""

def _extract_pdf_text_sync(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""

    # Prefer PyMuPDF (fitz) if available (typically better extraction).
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts: List[str] = []
        for page in doc:
            try:
                parts.append(page.get_text("text"))
            except Exception:
                continue
        text = "\n".join(parts)
        return text
    except Exception:
        pass

    # Fallback to pypdf.
    try:
        import io
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception:
        return ""

def _extract_text_from_zip_sync(zip_bytes: bytes) -> str:
    if not zip_bytes:
        return ""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n for n in zf.namelist() if n and not n.endswith("/")]
            if not names:
                return ""

            # Prefer PDFs, then HTML, then TXT/XML.
            def _rank(name: str) -> int:
                ln = name.lower()
                if ln.endswith(".pdf"):
                    return 0
                if ln.endswith(".htm") or ln.endswith(".html") or ln.endswith(".xhtml"):
                    return 1
                if ln.endswith(".txt") or ln.endswith(".xml"):
                    return 2
                return 3

            names_sorted = sorted(names, key=_rank)
            top = names_sorted[0]
            data = zf.read(top)

            ln = top.lower()
            if ln.endswith(".pdf"):
                return _extract_pdf_text_sync(data)
            if ln.endswith((".htm", ".html", ".xhtml")):
                return strip_html_to_text(_bytes_to_text_best_effort(data))
            # TXT/XML - treat as text and strip tags lightly.
            return strip_html_to_text(_bytes_to_text_best_effort(data))
    except Exception:
        return ""

async def _extract_text_from_bytes(url: str, content_type: str, data: bytes) -> str:
    # Offload CPU-heavy PDF/ZIP parsing.
    if _is_probably_pdf(url, content_type):
        return await asyncio.to_thread(_extract_pdf_text_sync, data)
    if _is_probably_zip(url, content_type):
        return await asyncio.to_thread(_extract_text_from_zip_sync, data)

    # Otherwise treat as text/HTML.
    txt = _bytes_to_text_best_effort(data)
    # Many pages are HTML even if content-type is text/plain.
    return strip_html_to_text(txt) if "<" in txt and ">" in txt else txt

class HttpDocumentTextAdapter(DocumentTextPort):
    def __init__(self, *, http: httpx.AsyncClient, config: RuntimeConfig):
        self._http = http
        self._config = config

        self._doc_cache = HttpConditionalHeadersCache(self._config.shared_state_dir / "http_doc_cache.json")
        self._requester = AsyncHttpRequester(
            http=self._http,
            cache=self._doc_cache,
            default_headers={},
            max_attempts=int((os.getenv("HTTP_MAX_ATTEMPTS") or "4").strip() or 4),
        )

    async def _fetch_url_bytes(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_conditional: bool = True,
        cache_drop_keys: Optional[set[str]] = None,
    ) -> Tuple[bytes, str]:
        url = (url or "").strip()
        if not url:
            return b"", ""

        cache_key = _http_cache_key(url, params, drop_keys=cache_drop_keys)
        try:
            resp = await self._requester.get(
                url,
                headers=headers,
                params=params,
                timeout=self._config.http_timeout_seconds,
                follow_redirects=True,
                use_conditional=use_conditional,
                cache_key=cache_key,
            )
        except Exception as e:
            logging.warning("[HTTP] fetch failed: %s", e)
            return b"", ""

        if resp is None and use_conditional:
            # Conditional GET hit (304) but we do not cache bodies; fall back to an unconditional fetch.
            try:
                resp = await self._requester.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self._config.http_timeout_seconds,
                    follow_redirects=True,
                    use_conditional=False,
                    cache_key=cache_key,
                )
            except Exception as e:
                logging.warning("[HTTP] unconditional refetch failed: %s", e)
                return b"", ""

        if resp is None:
            return b"", ""

        ct = resp.headers.get("Content-Type") or ""
        try:
            return resp.content, ct
        except Exception:
            try:
                return resp.text.encode("utf-8", errors="ignore"), ct
            except Exception:
                return b"", ct

    async def fetch_document_text(self, doc: RegulatoryDocumentHandle) -> str:
        # Allow ingestion adapters to provide a safe fallback snippet.
        fallback = ""
        try:
            if isinstance(doc.metadata, dict):
                fb = doc.metadata.get("raw_text") or doc.metadata.get("summary") or ""
                if isinstance(fb, str):
                    fallback = fb.strip()
        except Exception:
            fallback = ""

        try:
            data, ct = await self._fetch_url_bytes(doc.url, use_conditional=True)
            if not data:
                return fallback
            txt = await _extract_text_from_bytes(doc.url, ct, data)
            return _cap_text(txt) if txt.strip() else fallback
        except Exception as e:
            logging.warning("[%s] fetch failed: %s", doc.source, e)
            return fallback

