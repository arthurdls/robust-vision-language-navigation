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

  const formatTs = (mtime) => {
    if (!mtime) return "waiting...";
    const d = new Date(mtime * 1000);
    return d.toTimeString().slice(0, 8);
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

  const renderHeader = (state) => {
    if (state.active_subgoal && state.subgoals.length) {
      $("subgoal-index").textContent =
        `${state.active_subgoal.index} / ${state.subgoals.length}`;
      $("subgoal-name").textContent =
        state.active_subgoal.name.replace(/_/g, " ");
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
      const label = sg.name.replace(/_/g, " ");
      const short = label.length > 24 ? label.slice(0, 22) + "…" : label;
      return `<span class="crumb" data-state="${stateAttr}" title="${label}">`
           + `<span class="crumb-glyph">${glyph}</span>`
           + `<span class="crumb-label">${sg.index}. ${short}</span>`
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

  const renderConvergence = (state) => {
    const panel = $("panel-convergence");
    if (!state.convergence_label) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    renderImage("convergence", state.convergence_image_mtime);
    renderText("convergence", state.convergence_response,
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
    $("elapsed").textContent = formatElapsed(now - (runStartEpoch || now));

    renderHeader(state);
    renderBreadcrumb(state);
    renderImage("local",  state.local_image_mtime);
    renderImage("global", state.global_image_mtime);
    renderText("diary",    state.diary_text,    state.diary_mtime,    state.checkpoint_label);
    renderText("response", state.response_text, state.response_mtime, state.checkpoint_label);
    renderConvergence(state);
    setStatus(deriveStatus(state, now));
  };

  setInterval(tick, POLL_MS);
  tick();
})();
