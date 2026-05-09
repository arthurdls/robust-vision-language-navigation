window.hydrate = function(state) {
  // Topbar
  document.getElementById("bar-time").textContent = state.bar_time || "00:00 / 00:00";
  document.getElementById("bar-sg").textContent = String(state.subgoal_index || 1);
  document.getElementById("bar-sg-total").textContent = String(state.subgoal_total || 4);
  document.getElementById("bar-slug").textContent = state.subgoal_slug || "init";
  document.getElementById("bar-sg-text").textContent = state.subgoal_text ? '"' + state.subgoal_text + '"' : "";
  document.getElementById("bar-completion").textContent = (state.completion ?? 0).toFixed(2);
  document.getElementById("bar-drive").textContent = state.drive || "stop";

  // Local pane
  var localImg = document.getElementById("local-img");
  if (state.local_image) { localImg.src = "file://" + state.local_image; } else { localImg.src = ""; }
  document.getElementById("local-reasoning").textContent = state.local_reasoning || "awaiting first checkpoint...";

  // Global pane
  var globalImg = document.getElementById("global-img");
  if (state.global_image) { globalImg.src = "file://" + state.global_image; } else { globalImg.src = ""; }
  document.getElementById("global-reasoning").textContent = state.global_reasoning || "awaiting first checkpoint...";

  var meta = document.getElementById("global-meta");
  meta.innerHTML = "";
  (state.global_chips || []).forEach(function(c) {
    var span = document.createElement("span");
    span.className = "chip" + (c.kind ? " " + c.kind : "");
    span.textContent = c.label;
    meta.appendChild(span);
  });

  // Diary
  var diary = document.getElementById("diary-list");
  diary.innerHTML = "";
  var entries = state.diary_entries || [];
  entries.forEach(function(entry, i) {
    var li = document.createElement("li");
    if (entry.kind === "checkpoint") {
      li.innerHTML = entry.text;
      li.className = "checkpoint";
    } else {
      li.innerHTML = entry.text;
    }
    if (i === entries.length - 1 && entries.length > 0) {
      li.className += " newest";
    } else if (i < entries.length - 6) {
      li.className += " fading";
    }
    diary.appendChild(li);
  });

  // Convergence
  document.getElementById("conv-reasoning").textContent = state.conv_reasoning || "no convergence yet";
  var convMeta = document.getElementById("conv-meta");
  convMeta.innerHTML = "";
  (state.conv_chips || []).forEach(function(c) {
    var span = document.createElement("span");
    span.className = "chip" + (c.kind ? " " + c.kind : "");
    span.textContent = c.label;
    convMeta.appendChild(span);
  });
  var convStatus = document.getElementById("conv-status");
  if (state.conv_status) {
    convStatus.textContent = state.conv_status;
    if (state.conv_status === "complete") {
      convStatus.style.background = "#2a6a3a";
      convStatus.style.color = "#cfe6cf";
    } else {
      convStatus.style.background = "#6a4a2a";
      convStatus.style.color = "#fdcc88";
    }
  } else {
    convStatus.textContent = "idle";
    convStatus.style.background = "#6a4a2a";
    convStatus.style.color = "#fdcc88";
  }
};
