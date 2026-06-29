const THINKING_STEPS = [
  "Đang phân tích yêu cầu",
  "Đang tra cứu căn cứ pháp lý",
  "Đang đọc tài liệu liên quan",
  "Đang đối chiếu văn bản",
  "Đang lập luận pháp lý",
  "Đang chuẩn bị kết quả",
];

const STORAGE_KEY = "lexi_legal_conversations_v1";

let systemReady = false;
let activeConversationId = null;
let conversations = loadConversations();
let thinkingTimer = null;
let stepTimer = null;
let isSubmitting = false;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function uid() {
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function getActiveConversation() {
  return conversations.find((c) => c.id === activeConversationId) || null;
}

function formatSidebarDate(value) {
  if (!value) return "";
  const date = new Date(value);
  const diffMs = Date.now() - date.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffHours < 1) return "Vừa xong";
  if (diffHours < 24) return `${diffHours} giờ trước`;
  if (diffDays === 1) return "Hôm qua";
  if (diffDays < 7) return `${diffDays} ngày trước`;
  return date.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" });
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 192)}px`;
  const btn = textarea.closest(".lexi-prompt-box")?.querySelector(".lexi-send-round");
  if (btn) btn.classList.toggle("has-content", !!textarea.value.trim());
}

function showView(name) {
  $("#welcome-view").classList.toggle("hidden", name !== "welcome");
  $("#chat-view").classList.toggle("hidden", name !== "chat");
}

function setTab(tab) {
  $$(".lexi-nav-item[data-tab], .lexi-nav-chat-row").forEach((el) => {
    el.classList.toggle("active", el.dataset.tab === tab);
  });
  $$(".lexi-tab-panel").forEach((p) => p.classList.remove("active"));
  $(`#panel-${tab}`)?.classList.add("active");
}

function openSourcePanel(open = true) {
  $("#source-panel").dataset.open = open ? "true" : "false";
}

function scrollChatToBottom(smooth = true) {
  const scroller = $("#messages-scroll");
  if (!scroller) return;
  scroller.scrollTo({
    top: scroller.scrollHeight,
    behavior: smooth ? "smooth" : "auto",
  });
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text ?? "";
  return d.innerHTML;
}

function lexiIconHtml(spin = false, size = 36) {
  return `<img src="/static/lexi-icon.svg" alt="Lexi" class="lexi-avatar${spin ? " spin" : ""}" width="${size}" height="${size}" />`;
}

function renderPreResponse(steps, streaming = true) {
  const stepsHtml = steps
    .map(
      (label, i) => `
    <div class="lexi-timeline-item">
      <div class="lexi-timeline-dot ${i === steps.length - 1 && streaming ? "active" : "done"}">
        ${i === steps.length - 1 && streaming ? '<span class="inner law-breathe"></span>' : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg>'}
      </div>
      <span>${escapeHtml(label)}</span>
    </div>`,
    )
    .join("");

  return `
    <div class="lexi-pre-response" id="live-pre-response">
      <button type="button" class="lexi-pre-response-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
        <span class="${streaming ? "shimmer" : ""}">${streaming ? "Đang xử lý pháp lý" : `Hoàn tất ${steps.length} bước xử lý`}</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
      </button>
      <div class="lexi-pre-steps">${stepsHtml}</div>
    </div>`;
}

function renderMessages(history, options = {}) {
  const container = $("#messages");
  container.innerHTML = "";

  for (let i = 0; i < history.length; i++) {
    const msg = history[i];
    const wrap = document.createElement("div");
    wrap.className = `lexi-msg lexi-msg-${msg.role}`;

    if (msg.role === "user") {
      wrap.innerHTML = `<div class="lexi-bubble">${escapeHtml(msg.content)}</div>`;
    } else {
      const isLast = i === history.length - 1;
      wrap.innerHTML = `
        <div class="lexi-avatar-row">${lexiIconHtml(false)}</div>
        ${msg.preSteps?.length ? renderPreResponse(msg.preSteps, false) : ""}
        <div class="lexi-bubble lexi-markdown">${marked.parse(msg.content || "")}</div>
        <div class="lexi-copy-row">
          <button type="button" class="lexi-copy-btn" data-copy-idx="${i}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Sao chép
          </button>
        </div>`;
    }
    container.appendChild(wrap);
  }

  if (options.streaming) {
    const live = document.createElement("div");
    live.className = "lexi-msg lexi-msg-assistant";
    live.id = "streaming-assistant";
    live.innerHTML = `
      <div class="lexi-avatar-row">${lexiIconHtml(true)}</div>
      ${renderPreResponse(options.steps || [THINKING_STEPS[0]], true)}
    `;
    container.appendChild(live);
  }

  container.querySelectorAll(".lexi-copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const idx = Number(btn.dataset.copyIdx);
      const text = history[idx]?.content || "";
      await navigator.clipboard.writeText(text);
      const prev = btn.innerHTML;
      btn.textContent = "Đã sao chép";
      setTimeout(() => { btn.innerHTML = prev; }, 1500);
    });
  });

  scrollChatToBottom();
}

function renderSources(sources, markdown) {
  const panel = $("#sources-content");
  $("#source-tab-title").textContent = sources?.length
    ? `Căn cứ pháp lý (${sources.length})`
    : "Căn cứ pháp lý";

  if (!sources?.length) {
    panel.innerHTML = `<div class="lexi-markdown lexi-sources-empty">${marked.parse(markdown || "Không có tài liệu tham khảo.")}</div>`;
    openSourcePanel(!!markdown && markdown.includes("**"));
    return;
  }

  panel.innerHTML = sources
    .map(
      (doc, i) => `
    <article class="lexi-source-card" data-index="${i}">
      <h3>${escapeHtml(doc.law_id || "")} — ${escapeHtml(doc.title || "Không có tiêu đề")}</h3>
      <p>${escapeHtml(doc.excerpt || "")}</p>
    </article>`,
    )
    .join("");

  panel.querySelectorAll(".lexi-source-card").forEach((card) => {
    card.addEventListener("click", () => {
      const doc = sources[Number(card.dataset.index)];
      panel.innerHTML = `
        <button type="button" class="lexi-copy-btn" id="back-sources" style="margin-bottom:.75rem">← Quay lại danh sách</button>
        <div class="lexi-source-detail">
          <h4>${escapeHtml(doc.law_id)} — ${escapeHtml(doc.title)}</h4>
          <p>${escapeHtml(doc.excerpt)}</p>
        </div>`;
      $("#back-sources").addEventListener("click", () => renderSources(sources, markdown));
    });
  });

  openSourcePanel(true);
}

function renderConversationList() {
  const list = $("#conversation-list");
  if (!conversations.length) {
    list.innerHTML = `<p style="padding:.75rem 1.25rem;font-size:.75rem;color:#94a3b8">Chưa có hội thoại soạn thảo.</p>`;
    return;
  }
  list.innerHTML = conversations
    .slice()
    .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
    .map(
      (c) => `
    <div class="lexi-conv-item ${c.id === activeConversationId ? "active" : ""}" data-id="${c.id}">
      <button type="button" class="lexi-conv-btn">
        <span class="lexi-conv-title">${escapeHtml(c.title || "Cuộc trò chuyện chưa đặt tên")}</span>
        <span class="lexi-conv-date">${formatSidebarDate(c.updated_at)}</span>
      </button>
    </div>`,
    )
    .join("");

  list.querySelectorAll(".lexi-conv-item").forEach((item) => {
    item.querySelector(".lexi-conv-btn").addEventListener("click", () => {
      selectConversation(item.dataset.id);
    });
  });
}

function selectConversation(id) {
  activeConversationId = id;
  const conv = getActiveConversation();
  if (!conv) return;
  setTab("chat");
  if (conv.history?.length) {
    showView("chat");
    renderMessages(conv.history);
    renderSources(conv.sources, conv.sources_markdown);
  } else {
    showView("welcome");
  }
  renderConversationList();
}

function newConversation() {
  const id = uid();
  const conv = {
    id,
    title: "Cuộc trò chuyện mới",
    history: [],
    sources: [],
    sources_markdown: "",
    updated_at: new Date().toISOString(),
  };
  conversations.unshift(conv);
  activeConversationId = id;
  saveConversations();
  renderConversationList();
  showView("welcome");
  setTab("chat");
  $("#sources-content").innerHTML = '<p class="lexi-sources-empty">Tài liệu tham khảo sẽ hiển thị sau mỗi câu trả lời.</p>';
  openSourcePanel(false);
  playHeroAnimation();
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const init = data.initialization || {};
    const pill = $("#system-status");
    const text = pill.querySelector(".status-text");

    if (init.status === "ready") {
      systemReady = true;
      pill.className = "lexi-status-pill ready";
      text.textContent = "Sẵn sàng";
      return true;
    }
    if (init.status === "error") {
      systemReady = false;
      pill.className = "lexi-status-pill error";
      text.textContent = init.message || "Lỗi khởi tạo";
      return false;
    }
    systemReady = false;
    pill.className = "lexi-status-pill";
    text.textContent = `${init.progress || 0}% — ${init.details || init.message || "Đang tải..."}`;
    return false;
  } catch {
    return false;
  }
}

function startStatusPolling() {
  pollStatus();
  setInterval(pollStatus, 2000);
}

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const samples = await res.json();
    // Samples shown via sidebar history area on mobile could add chips — skip for fidelity
    void samples;
  } catch (e) {
    console.warn("samples", e);
  }
}

function cycleThinkingSteps(onStep) {
  let i = 0;
  clearInterval(stepTimer);
  onStep([THINKING_STEPS[0]]);
  stepTimer = setInterval(() => {
    i = Math.min(i + 1, THINKING_STEPS.length - 1);
    onStep(THINKING_STEPS.slice(0, i + 1));
  }, 1800);
}

async function submitMessage(message) {
  const text = (message || "").trim();
  if (!text || isSubmitting) return;

  if (!systemReady) {
    const pill = $("#system-status");
    pill.className = "lexi-status-pill";
    pill.querySelector(".status-text").textContent = "⏳ Đang khởi tạo — vui lòng đợi vài giây...";
    await pollStatus();
    if (!systemReady) return;
  }

  if (!activeConversationId) newConversation();
  const conv = getActiveConversation();

  isSubmitting = true;
  showView("chat");
  setTab("chat");

  const pendingHistory = [...(conv.history || []), { role: "user", content: text }];
  renderMessages(pendingHistory, { streaming: true, steps: [THINKING_STEPS[0]] });

  const boxes = $$(".lexi-prompt-box");
  boxes.forEach((b) => b.classList.add("loading"));
  $$(".lexi-send-round").forEach((b) => b.classList.add("loading"));

  cycleThinkingSteps((steps) => {
    const live = $("#streaming-assistant");
    if (!live) return;
    live.innerHTML = `
      <div class="lexi-avatar-row">${lexiIconHtml(true)}</div>
      ${renderPreResponse(steps, true)}`;
  });

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: conv.history || [],
        session_id: conv.id,
      }),
    });
    const data = await res.json();
    const lastAssistant = (data.history || []).filter((m) => m.role === "assistant").pop();
    if (lastAssistant && data.meta?.show_steps) {
      lastAssistant.preSteps = [...THINKING_STEPS];
    }

    conv.history = data.history || conv.history;
    conv.sources = data.sources || [];
    conv.sources_markdown = data.sources_markdown || "";
    conv.updated_at = new Date().toISOString();
    if (conv.title === "Cuộc trò chuyện mới") {
      conv.title = text.length > 48 ? `${text.slice(0, 48)}...` : text;
    }
    saveConversations();
    renderConversationList();
    renderMessages(conv.history);
    renderSources(conv.sources, conv.sources_markdown);
  } catch (err) {
    conv.history = [
      ...(conv.history || []),
      { role: "user", content: text },
      { role: "assistant", content: `❌ Lỗi: ${err.message}`, preSteps: THINKING_STEPS },
    ];
    saveConversations();
    renderMessages(conv.history);
  } finally {
    clearInterval(stepTimer);
    isSubmitting = false;
    boxes.forEach((b) => b.classList.remove("loading"));
    $$(".lexi-send-round").forEach((b) => b.classList.remove("loading"));
    ["#input-welcome", "#input-chat"].forEach((sel) => {
      const el = $(sel);
      if (el) {
        el.value = "";
        autoResize(el);
      }
    });
  }
}

function wireComposer(textarea, sendBtn) {
  const sync = () => autoResize(textarea);
  textarea.addEventListener("input", sync);
  textarea.addEventListener("keyup", sync);
  textarea.addEventListener("change", sync);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (textarea.value.trim()) submitMessage(textarea.value);
    }
  });
  sendBtn.addEventListener("click", () => {
    if (textarea.value.trim()) submitMessage(textarea.value);
  });
  sync();
}

function playHeroAnimation() {
  const stage = $("#hero-stage");
  stage.classList.remove("loaded");
  requestAnimationFrame(() => {
    setTimeout(() => stage.classList.add("loaded"), 80);
  });
}

function setupSourceResize() {
  const panel = $("#source-panel");
  const handle = $("#source-resize");
  let startX = 0;
  let startW = 0;

  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startX = e.clientX;
    startW = panel.offsetWidth;
    const onMove = (ev) => {
      const delta = startX - ev.clientX;
      const next = Math.min(window.innerWidth * 0.55, Math.max(320, startW + delta));
      document.documentElement.style.setProperty("--source-w", `${next}px`);
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });
}

async function boot() {
  wireComposer($("#input-welcome"), $("#send-welcome"));
  wireComposer($("#input-chat"), $("#send-chat"));

  $("#btn-new-chat").addEventListener("click", newConversation);
  $("#nav-chat").addEventListener("click", () => setTab("chat"));
  $("#nav-workflows").addEventListener("click", () => setTab("workflows"));
  $("#nav-documents").addEventListener("click", () => setTab("documents"));

  $("#btn-toggle-sidebar").addEventListener("click", () => {
    const sb = $("#sidebar");
    sb.dataset.open = sb.dataset.open === "true" ? "false" : "true";
  });

  $("#history-toggle").addEventListener("click", () => {
    $("#history-toggle").classList.toggle("collapsed");
    $("#conversation-list").classList.toggle("collapsed");
  });

  $("#close-source-panel").addEventListener("click", () => openSourcePanel(false));
  $("#close-source-tab").addEventListener("click", () => openSourcePanel(false));

  setupSourceResize();
  playHeroAnimation();
  renderConversationList();
  startStatusPolling();
  loadSamples();
}

boot();
