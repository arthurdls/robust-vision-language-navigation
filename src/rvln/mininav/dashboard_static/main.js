(() => {
  const POLL_MS = 250;
  const WAITING_THRESHOLD_S = 8;
  const CONVERGING_WINDOW_S = 30;
  const ERROR_AFTER_FAILURES = 3;

  const $ = (id) => document.getElementById(id);

  let runStartEpoch = null;
  let consecutiveFailures = 0;
  const lastImageMtime = { local: 0, global: 0, convergence: 0 };
  const lastText = { diary: null, response: null, convergence: null };

  const formatElapsed = (s) => {
    s = Math.max(0, Math.floor(s));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return h > 0
      ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
      : `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };

  const formatAge = (mtime, now) => {
    if (!mtime) return "waiting...";
    const ref = (typeof now === "number" && now > 0) ? now : (Date.now() / 1000);
    const ageS = Math.max(0, Math.floor(ref - mtime));
    if (ageS < 1)    return "just now";
    if (ageS < 60)   return `${ageS}s ago`;
    if (ageS < 3600) {
      const m = Math.floor(ageS / 60);
      const s = ageS % 60;
      return s ? `${m}m ${s}s ago` : `${m}m ago`;
    }
    const h = Math.floor(ageS / 3600);
    const m = Math.floor((ageS % 3600) / 60);
    return m ? `${h}h ${m}m ago` : `${h}h ago`;
  };

  // Wall-clock "now" passed in from the tick so all panels share one reference.
  let renderNow = Date.now() / 1000;
  const formatTs = (mtime) => formatAge(mtime, renderNow);

  const prettyJsonIfPossible = (s) => {
    if (!s) return s;
    const trimmed = s.trim();
    if (!trimmed || (trimmed[0] !== "{" && trimmed[0] !== "[")) return s;
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2);
    } catch {
      return s;
    }
  };

  const deriveStatus = (state, now) => {
    if (state.run_complete) return "done";
    const newest = Math.max(
      state.local_image_mtime || 0,
      state.global_image_mtime || 0,
      state.convergence_image_mtime || 0,
      state.diary_mtime || 0,
      state.response_mtime || 0,
    );
    const cvAge = state.convergence_image_mtime
      ? now - state.convergence_image_mtime : Infinity;
    const cpAge = Math.max(state.local_image_mtime || 0,
                           state.global_image_mtime || 0);
    if (state.convergence_image_mtime && cvAge < CONVERGING_WINDOW_S
        && state.convergence_image_mtime > cpAge) {
      return "converging";
    }
    if (newest && (now - newest) > WAITING_THRESHOLD_S) return "waiting";
    if (newest === 0) return "waiting";
    return "running";
  };

  const setStatus = (status) => {
    const pill = $("status-pill");
    pill.dataset.status = status;
    pill.textContent = {
      running:    "RUNNING",
      waiting:    "WAITING",
      converging: "CONVERGING",
      done:       "DONE",
      error:      "ERROR",
    }[status] || status.toUpperCase();
  };

  const subgoalText = (sg) =>
    sg.label || sg.name.replace(/_/g, " ");

  const renderHeader = (state) => {
    if (state.active_subgoal && state.subgoals.length) {
      $("subgoal-index").textContent =
        `${state.active_subgoal.index} / ${state.subgoals.length}`;
      $("subgoal-name").textContent = subgoalText(state.active_subgoal);
    } else {
      $("subgoal-index").textContent = "--/--";
      $("subgoal-name").textContent = "preparing run...";
    }
    $("checkpoint-count").textContent =
      state.checkpoint_count ? String(state.checkpoint_count) : "--";
  };

  const renderBreadcrumb = (state) => {
    const nav = $("breadcrumb");
    const activeIdx = state.active_subgoal ? state.active_subgoal.index : -1;
    const wantHTML = state.subgoals.map((sg) => {
      const completed = sg.index < activeIdx;
      const active = sg.index === activeIdx;
      const stateAttr = active ? "active" : completed ? "completed" : "future";
      const glyph = active ? "▸" : completed ? "◉" : "○";
      const label = subgoalText(sg);
      const display = active || label.length <= 24 ? label : label.slice(0, 22) + "…";
      return `<span class="crumb" data-state="${stateAttr}" title="${label}">`
           + `<span class="crumb-glyph">${glyph}</span>`
           + `<span class="crumb-label">${sg.index}. ${display}</span>`
           + `</span>`;
    }).join("");
    if (nav.innerHTML !== wantHTML) {
      nav.innerHTML = wantHTML;
      const active = nav.querySelector('.crumb[data-state="active"]');
      if (active) active.scrollIntoView({ behavior: "smooth", inline: "center" });
    }
  };

  const renderImage = (slot, mtime) => {
    const img = $(`img-${slot}`);
    const empty = $(`${slot}-empty`);
    const ts = $(`${slot}-ts`);
    if (mtime && mtime !== lastImageMtime[slot]) {
      lastImageMtime[slot] = mtime;
      img.src = `/img/${slot}?t=${mtime}`;
      img.hidden = false;
      if (empty) empty.hidden = true;
    }
    if (ts) ts.textContent = formatTs(mtime);
  };

  const renderText = (slot, content, mtime, label) => {
    const body = $(`${slot}-body`);
    const ts = $(`${slot}-ts`);
    if (content !== lastText[slot]) {
      lastText[slot] = content;
      const wasAtBottom = body.scrollTop + body.clientHeight
                          >= body.scrollHeight - 4;
      body.classList.add("fading");
      setTimeout(() => {
        if (content) {
          body.textContent = content;
        } else {
          body.innerHTML = '<span class="text-empty">no entries yet</span>';
        }
        body.classList.remove("fading");
        if (wasAtBottom) body.scrollTop = body.scrollHeight;
      }, 90);
    }
    if (ts) ts.textContent = label
      ? `${label} @ ${formatTs(mtime)}`
      : formatTs(mtime);
  };

  const escapeHTML = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));

  const lastChainKey = { value: null };

  const renderInstructionChain = (state) => {
    const body = $("actions-body");
    const ts = $("actions-ts");
    const chain = Array.isArray(state.instruction_chain)
      ? state.instruction_chain : [];
    const sg = state.active_subgoal;
    if (chain.length === 0) {
      body.innerHTML = '<span class="text-empty">no instruction yet</span>';
      ts.textContent = sg ? `subgoal ${sg.index}` : "waiting...";
      lastChainKey.value = "";
      return;
    }
    const last = chain[chain.length - 1];
    const lastAge = formatTs(last && last.mtime);
    const subgoalLabel = sg ? `subgoal ${sg.index} • ` : "";
    ts.textContent = `${subgoalLabel}${chain.length} step${chain.length === 1 ? "" : "s"} • last change ${lastAge}`;

    // Structural cache: ignore mtime so the row body isn't rebuilt every
    // tick just because the relative-age clock advanced. The header ts above
    // already updates every tick.
    const structural = chain.map((e) => ({
      l: e.label, s: e.source, i: e.instruction,
      d: e.diagnosis || "", r: e.reasoning || "",
    }));
    const key = JSON.stringify({ i: sg ? sg.index : null, c: structural });
    if (key === lastChainKey.value) return;
    lastChainKey.value = key;

    const rows = chain.map((e, i) => {
      const isActive = i === chain.length - 1;
      const badge = e.source === "initial" ? "init" : (e.label || "?");
      const glyph = isActive ? "▸" : (e.source === "initial" ? "•" : "·");
      const diag = e.diagnosis
        ? `<span class="action-diag">${escapeHTML(e.diagnosis)}</span>` : "";
      const reasoning = e.reasoning
        ? `<div class="action-reasoning">${escapeHTML(e.reasoning)}</div>` : "";
      const nowChip = isActive
        ? '<span class="action-now">now executing</span>' : "";
      const cls = "action-row" + (isActive ? " action-row-active" : "")
                + (e.source === "initial" ? " action-row-initial" : "");
      return `<div class="${cls}">`
           + `<span class="action-glyph">${glyph}</span>`
           + `<span class="action-badge">${escapeHTML(badge)}</span>`
           + `<div class="action-text">`
           +   `<div class="action-instruction-row">`
           +     `<span class="action-instruction">${escapeHTML(e.instruction)}</span>`
           +     nowChip
           +   `</div>`
           +   diag
           +   reasoning
           + `</div>`
           + `</div>`;
    }).join("");
    body.innerHTML = rows;
    const active = body.querySelector(".action-row-active");
    if (active) active.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  const renderConvergence = (state) => {
    const panel = $("panel-convergence");
    if (!state.convergence_label) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    renderImage("convergence", state.convergence_image_mtime);
    renderText("convergence", prettyJsonIfPossible(state.convergence_response),
               state.convergence_image_mtime, state.convergence_label);
  };

  const tick = async () => {
    let state = null;
    try {
      const resp = await fetch("/api/state", { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state = await resp.json();
      consecutiveFailures = 0;
    } catch (err) {
      consecutiveFailures++;
      if (consecutiveFailures >= ERROR_AFTER_FAILURES) setStatus("error");
      return;
    }

    if (runStartEpoch === null && state.server_time) {
      runStartEpoch = state.server_time;
    }
    const now = state.server_time || (Date.now() / 1000);
    renderNow = now;
    $("elapsed").textContent = formatElapsed(now - (runStartEpoch || now));

    renderHeader(state);
    renderBreadcrumb(state);
    renderImage("local",  state.local_image_mtime);
    renderImage("global", state.global_image_mtime);
    renderText("diary",    state.diary_text,    state.diary_mtime,    state.checkpoint_label);
    renderText("response", prettyJsonIfPossible(state.response_text), state.response_mtime, state.checkpoint_label);
    renderInstructionChain(state);
    renderConvergence(state);
    setStatus(deriveStatus(state, now));
  };

  setInterval(tick, POLL_MS);
  tick();
})();
