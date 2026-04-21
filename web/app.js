// Regfeed dashboard — vanilla JS
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};

function fmtAge(hours) {
  if (hours == null) return "never";
  if (hours < 1) return `${Math.round(hours * 60)}m ago`;
  if (hours < 48) return `${hours.toFixed(1)}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function fmtDate(iso) {
  if (!iso) return "—";
  try { return new Date(iso).toLocaleString(); }
  catch { return iso; }
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ── Sources ───────────────────────────────────────────────────────────
async function renderSources() {
  const container = $("#sources");
  container.innerHTML = "<div class='muted'>loading…</div>";
  try {
    const { sources } = await getJSON("/gui/sources");
    container.innerHTML = "";
    for (const s of sources) {
      const tile = el("div", "tile");
      tile.dataset.source = s.source;
      tile.innerHTML = `
        <div class="head">
          <span class="dot ${s.status}"></span>
          <span class="name">${s.source.replace("_", " ")}</span>
        </div>
        <div class="meta">
          <div>Last data: <b>${fmtAge(s.age_hours)}</b></div>
          <div>Items: <b>${s.total}</b> · Published: <b>${s.published}</b></div>
          <div>Last publish: <b>${s.last_publish_at ? fmtDate(s.last_publish_at) : "—"}</b></div>
        </div>`;
      tile.addEventListener("click", () => loadMessages(s.source, tile));
      container.appendChild(tile);
    }
  } catch (e) {
    container.innerHTML = `<div class='muted'>Error: ${e.message}</div>`;
  }
}

// ── Channels ──────────────────────────────────────────────────────────
async function renderChannels() {
  const container = $("#channels");
  container.innerHTML = "<div class='muted'>loading…</div>";
  try {
    const { channels } = await getJSON("/gui/channels");
    container.innerHTML = "";
    for (const c of channels) {
      const ok = c.configured && !c.error;
      const dot = ok ? "green" : (c.configured ? "amber" : "red");
      const badge = c.paid ? "paid" : "free";
      const tile = el("div", "tile");
      tile.innerHTML = `
        <div class="head">
          <span class="dot ${dot}"></span>
          <span class="name">${c.label}</span>
          <span class="badge ${badge}">${badge}</span>
        </div>
        <div class="meta">
          <div>Chat: <b>${c.title || c.chat_id || "not configured"}</b></div>
          <div>Subscribers: <b>${c.member_count ?? "—"}</b></div>
          <div>Type: <b>${c.type || "—"}</b></div>
          ${c.error ? `<div style="color:var(--red)">${c.error}</div>` : ""}
        </div>`;
      container.appendChild(tile);
    }
  } catch (e) {
    container.innerHTML = `<div class='muted'>Error: ${e.message}</div>`;
  }
}

// ── Messages ──────────────────────────────────────────────────────────
async function loadMessages(source, tile) {
  document.querySelectorAll(".tile.active").forEach(t => t.classList.remove("active"));
  if (tile) tile.classList.add("active");

  $("#messages-card").hidden = false;
  $("#messages-source").textContent = source;
  const list = $("#messages");
  list.innerHTML = "<li class='muted'>loading…</li>";

  try {
    const { messages } = await getJSON(`/gui/sources/${source}/messages?limit=20`);
    if (!messages.length) {
      list.innerHTML = "<li class='muted'>No messages have been published from this feed yet.</li>";
      return;
    }
    list.innerHTML = "";
    for (const m of messages) {
      const li = el("li");
      const tierPill = m.tier
        ? `<span class="tier-pill ${m.tier}">${m.tier}</span>`
        : "";
      const ticker = m.ticker ? `<b>${m.ticker}</b> · ` : "";
      li.innerHTML = `
        <div class="title">
          ${ticker}${m.title || "(no title)"}${tierPill}
        </div>
        <div class="sub">
          Sent: ${fmtDate(m.telegram_sent_at)}
          ${m.confidence != null ? ` · Conf ${m.confidence}%` : ""}
          ${m.impact_score != null ? ` · Impact ${m.impact_score}` : ""}
          ${m.event_type ? ` · ${m.event_type}` : ""}
          ${m.url ? ` · <a href="${m.url}" target="_blank" rel="noopener">source</a>` : ""}
        </div>
        ${m.snippet ? `<div class="snippet">${m.snippet}</div>` : ""}`;
      list.appendChild(li);
    }
  } catch (e) {
    list.innerHTML = `<li class='muted'>Error: ${e.message}</li>`;
  }
}

// ── Refresh ───────────────────────────────────────────────────────────
function updateClock() {
  $("#clock").textContent = new Date().toLocaleTimeString();
}

async function refresh() {
  await Promise.all([renderSources(), renderChannels()]);
}

updateClock();
setInterval(updateClock, 1000);
refresh();
setInterval(refresh, 30_000);  // 30-second auto-refresh
