"""Tests for fulfillment.py — the Stripe → Telegram-invite + welcome-email
money path.

Everything external is faked:
  - Telegram createChatInviteLink → respx-mocked httpx route
  - SMTP → mocked smtplib
  - subscribers.db → temp-file SubscriberStore fixture (conftest)
  - ops alerts → monkeypatched send_ops_alert (and the autouse conftest
    fixture guarantees TELEGRAM_OPS_* are unset)

NOTE: importing fulfillment runs load_dotenv on the real Regfeed/.env, so
os.environ may hold real SMTP creds during the test run — every test that
goes near send_email/_smtp_config overrides those vars explicitly.
"""
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from test_helpers import log_test_context

import fulfillment
from fulfillment import (
    INVITE_TTL_SECONDS,
    MAX_ATTEMPTS,
    SiteConfig,
    _alert_delivery_failures,
    _smtp_config,
    count_exhausted,
    make_invite_link,
    process_site,
    render_email,
    send_email,
)


def _test_site(db_path: str) -> SiteConfig:
    """SiteConfig pointing at a temp DB and TEST_* env names so we can
    never pick up the real bot tokens from the loaded .env."""
    return SiteConfig(
        site_id="sec",
        db_path=db_path,
        bot_token_env="TEST_FULFILL_BOT_TOKEN",
        chat_id_env="TEST_FULFILL_CHAT_ID",
        product_name="Catalyst Wire SEC",
        product_url="https://sec.catalystwire.org",
        short_pitch="Test pitch.",
    )


@pytest.fixture
def site(subscribers_store, monkeypatch):
    monkeypatch.setenv("TEST_FULFILL_BOT_TOKEN", "1234:test-token")
    monkeypatch.setenv("TEST_FULFILL_CHAT_ID", "-100777")
    return _test_site(subscribers_store.db_path)


# ── make_invite_link ─────────────────────────────────────────────────────────

@respx.mock
def test_make_invite_link_happy_path():
    log_test_context("invite_link_happy")
    route = respx.post(
        "https://api.telegram.org/bot1234:test-token/createChatInviteLink"
    ).mock(return_value=httpx.Response(200, json={
        "ok": True, "result": {"invite_link": "https://t.me/+abc123"},
    }))

    before = int(time.time())
    link = make_invite_link(bot_token="1234:test-token", chat_id="-100777")
    assert link == "https://t.me/+abc123"

    sent = route.calls.last.request
    import json
    payload = json.loads(sent.content)
    # One-seat, no-join-request, 24h-TTL invite — the documented pattern.
    assert payload["chat_id"] == "-100777"
    assert payload["member_limit"] == 1
    assert payload["creates_join_request"] is False
    assert before + INVITE_TTL_SECONDS <= payload["expire_date"] <= before + INVITE_TTL_SECONDS + 5


@respx.mock
def test_make_invite_link_telegram_says_not_ok():
    respx.post(
        "https://api.telegram.org/bot1234:test-token/createChatInviteLink"
    ).mock(return_value=httpx.Response(200, json={
        "ok": False, "description": "Bad Request: not enough rights",
    }))
    with pytest.raises(RuntimeError, match="createChatInviteLink failed"):
        make_invite_link(bot_token="1234:test-token", chat_id="-100777")


@respx.mock
def test_make_invite_link_http_error_propagates():
    respx.post(
        "https://api.telegram.org/bot1234:test-token/createChatInviteLink"
    ).mock(return_value=httpx.Response(403, json={"ok": False}))
    with pytest.raises(httpx.HTTPStatusError):
        make_invite_link(bot_token="1234:test-token", chat_id="-100777")


# ── _smtp_config / send_email ────────────────────────────────────────────────

def test_smtp_config_missing_creds_raises(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "")
    monkeypatch.setenv("SMTP_PASSWORD", "")
    with pytest.raises(RuntimeError, match="SMTP not configured"):
        _smtp_config()


def test_smtp_config_defaults(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "ops@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_FROM_ADDRESS", raising=False)
    monkeypatch.delenv("SMTP_FROM_NAME", raising=False)
    cfg = _smtp_config()
    assert cfg["host"] == "smtp.gmail.com"
    assert cfg["port"] == 465
    assert cfg["from_addr"] == "ops@example.com"   # falls back to user
    assert cfg["from_name"] == "Catalyst Wire"


def _fake_smtp_env(monkeypatch, port):
    monkeypatch.setenv("SMTP_USER", "ops@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("SMTP_HOST", "smtp.test.invalid")
    monkeypatch.setenv("SMTP_PORT", str(port))
    monkeypatch.setenv("SMTP_FROM_ADDRESS", "hello@catalystwire.org")
    monkeypatch.setenv("SMTP_FROM_NAME", "Catalyst Wire")


def test_send_email_uses_implicit_tls_on_465(monkeypatch):
    log_test_context("send_email_465")
    _fake_smtp_env(monkeypatch, 465)
    with patch("fulfillment.smtplib.SMTP_SSL") as ssl_cls, \
         patch("fulfillment.smtplib.SMTP") as plain_cls:
        smtp = ssl_cls.return_value.__enter__.return_value
        send_email(to="buyer@example.com", subject="s", html="<p>h</p>", text="t")
        ssl_cls.assert_called_once()
        plain_cls.assert_not_called()
        smtp.login.assert_called_once_with("ops@example.com", "app-password")
        msg = smtp.send_message.call_args[0][0]
        assert msg["To"] == "buyer@example.com"
        assert "hello@catalystwire.org" in msg["From"]


def test_send_email_uses_starttls_on_587(monkeypatch):
    log_test_context("send_email_587")
    _fake_smtp_env(monkeypatch, 587)
    with patch("fulfillment.smtplib.SMTP_SSL") as ssl_cls, \
         patch("fulfillment.smtplib.SMTP") as plain_cls:
        smtp = plain_cls.return_value.__enter__.return_value
        send_email(to="buyer@example.com", subject="s", html="<p>h</p>", text="t")
        ssl_cls.assert_not_called()
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("ops@example.com", "app-password")
        smtp.send_message.assert_called_once()


# ── render_email ─────────────────────────────────────────────────────────────

def test_render_email_contains_invite_and_product(site):
    html, plain = render_email(site, "https://t.me/+abc123")
    for body in (html, plain):
        assert "https://t.me/+abc123" in body
        assert "Catalyst Wire SEC" in body
        assert "/mykey" in body
        assert "24 hours" in body
    assert site.product_url in html
    assert site.product_url in plain


# ── process_site ─────────────────────────────────────────────────────────────

def test_process_site_skips_when_db_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_FULFILL_BOT_TOKEN", "x")
    monkeypatch.setenv("TEST_FULFILL_CHAT_ID", "y")
    site = _test_site(str(tmp_path / "does-not-exist.db"))
    assert process_site(site) == {"site": "sec", "skipped": "db-missing"}


def test_process_site_skips_when_config_missing(subscribers_store, monkeypatch):
    monkeypatch.delenv("TEST_FULFILL_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TEST_FULFILL_CHAT_ID", raising=False)
    site = _test_site(subscribers_store.db_path)
    assert process_site(site) == {"site": "sec", "skipped": "config-missing"}


def test_process_site_nothing_to_deliver(site):
    assert process_site(site) == {"site": "sec", "delivered": 0, "failed": 0}


def test_process_site_delivers_and_marks(site, subscribers_store,
                                          add_subscriber, monkeypatch):
    log_test_context("process_site_happy")
    sub_id = add_subscriber(email="Buyer@Example.com")
    sent = []
    monkeypatch.setattr(fulfillment, "make_invite_link",
                        lambda **kw: "https://t.me/+fake")
    monkeypatch.setattr(fulfillment, "send_email",
                        lambda **kw: sent.append(kw))

    result = process_site(site)
    assert result == {"site": "sec", "delivered": 1, "failed": 0}
    assert len(sent) == 1
    assert sent[0]["to"] == "buyer@example.com"
    assert "Catalyst Wire SEC" in sent[0]["subject"]
    assert "https://t.me/+fake" in sent[0]["html"]

    with subscribers_store.connect() as conn:
        row = conn.execute(
            "SELECT delivered_at, delivery_error FROM subscribers WHERE id=?",
            (sub_id,),
        ).fetchone()
    assert row["delivered_at"] is not None
    assert row["delivery_error"] is None


def test_process_site_is_idempotent(site, add_subscriber, monkeypatch):
    """Second run after a successful delivery must be a no-op (safe on a
    60s timer)."""
    add_subscriber()
    sent = []
    monkeypatch.setattr(fulfillment, "make_invite_link",
                        lambda **kw: "https://t.me/+fake")
    monkeypatch.setattr(fulfillment, "send_email",
                        lambda **kw: sent.append(kw))

    assert process_site(site)["delivered"] == 1
    assert process_site(site) == {"site": "sec", "delivered": 0, "failed": 0}
    assert len(sent) == 1   # exactly one email ever


def test_process_site_failure_records_error_and_alerts(
        site, subscribers_store, add_subscriber, monkeypatch):
    log_test_context("process_site_failure")
    sub_id = add_subscriber()
    alerts = []
    monkeypatch.setattr(fulfillment, "make_invite_link",
                        MagicMock(side_effect=RuntimeError("Telegram down")))
    monkeypatch.setattr(fulfillment, "send_email",
                        MagicMock(side_effect=AssertionError("must not email")))
    monkeypatch.setattr(fulfillment, "send_ops_alert",
                        lambda text: alerts.append(text))

    result = process_site(site)
    assert result == {"site": "sec", "delivered": 0, "failed": 1}

    with subscribers_store.connect() as conn:
        row = conn.execute(
            "SELECT delivered_at, delivery_attempts, delivery_error "
            "FROM subscribers WHERE id=?", (sub_id,),
        ).fetchone()
    assert row["delivered_at"] is None
    assert row["delivery_attempts"] == 1
    assert "Telegram down" in row["delivery_error"]

    assert len(alerts) == 1
    assert "FULFILLMENT FAILURE" in alerts[0]
    assert "[sec]" in alerts[0]


def test_process_site_dry_run_writes_nothing(site, subscribers_store,
                                              add_subscriber, monkeypatch):
    sub_id = add_subscriber()
    emails = MagicMock()
    monkeypatch.setattr(fulfillment, "make_invite_link",
                        lambda **kw: "https://t.me/+fake")
    monkeypatch.setattr(fulfillment, "send_email", emails)

    result = process_site(site, dry_run=True)
    assert result == {"site": "sec", "delivered": 1, "failed": 0}
    emails.assert_not_called()
    with subscribers_store.connect() as conn:
        row = conn.execute(
            "SELECT delivered_at, delivery_attempts FROM subscribers WHERE id=?",
            (sub_id,),
        ).fetchone()
    assert row["delivered_at"] is None       # dry run: no DB writes
    assert row["delivery_attempts"] == 0


def test_process_site_dry_run_failure_does_not_record(site, subscribers_store,
                                                       add_subscriber, monkeypatch):
    sub_id = add_subscriber()
    alerts = MagicMock()
    monkeypatch.setattr(fulfillment, "make_invite_link",
                        MagicMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(fulfillment, "send_ops_alert", alerts)

    result = process_site(site, dry_run=True)
    assert result == {"site": "sec", "delivered": 0, "failed": 1}
    alerts.assert_not_called()               # dry run: no ops alert either
    with subscribers_store.connect() as conn:
        attempts = conn.execute(
            "SELECT delivery_attempts FROM subscribers WHERE id=?", (sub_id,),
        ).fetchone()[0]
    assert attempts == 0


def test_exhausted_rows_are_excluded_from_retries(site, subscribers_store,
                                                   add_subscriber, monkeypatch):
    """A row that failed MAX_ATTEMPTS times is silently skipped by
    list_undelivered — this is the 'money path goes silent' hole that
    count_exhausted + the ops alert exist to surface (finding #3)."""
    log_test_context("exhausted_excluded")
    add_subscriber(delivery_attempts=MAX_ATTEMPTS)
    invite = MagicMock()
    monkeypatch.setattr(fulfillment, "make_invite_link", invite)

    result = process_site(site)
    assert result == {"site": "sec", "delivered": 0, "failed": 0}
    invite.assert_not_called()
    assert count_exhausted(subscribers_store) == 1


def test_count_exhausted_ignores_delivered_and_fresh_rows(subscribers_store,
                                                           add_subscriber):
    add_subscriber(email="a@x.com", subscription_id="sub_a",
                   delivery_attempts=MAX_ATTEMPTS)               # exhausted
    add_subscriber(email="b@x.com", subscription_id="sub_b",
                   delivery_attempts=2)                          # still retrying
    add_subscriber(email="c@x.com", subscription_id="sub_c",
                   delivery_attempts=MAX_ATTEMPTS, delivered=True)  # delivered anyway
    assert count_exhausted(subscribers_store) == 1


def test_count_exhausted_returns_zero_on_error():
    broken = MagicMock()
    broken.connect.side_effect = RuntimeError("db locked")
    assert count_exhausted(broken) == 0


def test_alert_message_mentions_exhausted_rows(site, subscribers_store,
                                                add_subscriber, monkeypatch):
    add_subscriber(delivery_attempts=MAX_ATTEMPTS)
    alerts = []
    monkeypatch.setattr(fulfillment, "send_ops_alert",
                        lambda text: alerts.append(text))
    _alert_delivery_failures(site, subscribers_store, failed=2)
    assert len(alerts) == 1
    assert "2 delivery attempt(s) failed" in alerts[0]
    assert "EXHAUSTED" in alerts[0]
    assert str(MAX_ATTEMPTS) in alerts[0]


def test_alert_silently_noops_without_ops_env(site, subscribers_store,
                                               add_subscriber):
    """With TELEGRAM_OPS_* unset (autouse fixture), the real send_ops_alert
    must no-op without any HTTP call — never falls back to product channels."""
    add_subscriber(delivery_attempts=MAX_ATTEMPTS)
    with patch("ops_alerts.httpx.post") as post:
        _alert_delivery_failures(site, subscribers_store, failed=1)
        post.assert_not_called()
