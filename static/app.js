/* People Counter front-end: upload -> place line -> poll job -> show results. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    video: null,                                   // upload metadata
    line: { x1: 1, y1: 0.55, x2: 0, y2: 0.55, inside_is_left: true },
    jobId: null,
    poll: null,
  };

  /* ---------------------------------------------------------- upload */
  const dropzone = $("dropzone");
  const fileInput = $("file-input");

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadVideo(file);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadVideo(fileInput.files[0]);
  });

  function uploadVideo(file) {
    hide($("upload-error"));
    const bar = $("upload-progress");
    show(bar);
    setBar(bar, 0, `Uploading ${file.name}…`);

    const body = new FormData();
    body.append("video", file);

    // XHR rather than fetch: it reports upload progress, which matters for
    // multi-hundred-megabyte clips.
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const pct = e.loaded / e.total;
        setBar(bar, pct, `Uploading ${file.name}… ${Math.round(pct * 100)}%`);
      }
    });
    xhr.addEventListener("load", () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status >= 400) {
        hide(bar);
        fail($("upload-error"), data.error || `Upload failed (HTTP ${xhr.status}).`);
        return;
      }
      setBar(bar, 1, `${data.original_name} — ready`);
      onUploaded(data);
    });
    xhr.addEventListener("error", () => {
      hide(bar);
      fail($("upload-error"), "Upload failed — is the server still running?");
    });
    xhr.send(body);
  }

  function onUploaded(data) {
    state.video = data;
    state.jobId = null;
    hide($("step-results"));
    hide($("job-error"));
    hide($("job-progress"));

    const preview = $("preview");
    preview.onload = () => { sizeOverlay(); drawLine(); };
    preview.src = `/api/preview/${data.video_id}?t=${Date.now()}`;

    const dur = data.duration ? `${data.duration.toFixed(1)}s` : "unknown length";
    $("video-meta").textContent =
      `${data.width}×${data.height} · ${data.fps.toFixed(1)} fps · ${dur}`;

    show($("step-line"));
    show($("step-settings"));
    $("step-line").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ------------------------------------------------------ line editor */
  const overlay = $("overlay");

  function sizeOverlay() {
    const img = $("preview");
    overlay.setAttribute("viewBox", `0 0 ${img.clientWidth} ${img.clientHeight}`);
  }
  window.addEventListener("resize", () => { sizeOverlay(); drawLine(); });

  function drawLine() {
    const img = $("preview");
    const w = img.clientWidth || 1;
    const h = img.clientHeight || 1;
    const ax = state.line.x1 * w, ay = state.line.y1 * h;
    const bx = state.line.x2 * w, by = state.line.y2 * h;

    for (const id of ["line-shadow", "line-main"]) {
      const el = $(id);
      el.setAttribute("x1", ax); el.setAttribute("y1", ay);
      el.setAttribute("x2", bx); el.setAttribute("y2", by);
    }
    $("handle-a").setAttribute("cx", ax); $("handle-a").setAttribute("cy", ay);
    $("handle-b").setAttribute("cx", bx); $("handle-b").setAttribute("cy", by);

    // Arrow points into the half-plane counted as "inside". Image y grows
    // downward, so the left-hand normal of (dx, dy) is (-dy, dx).
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    let nx = -dy / len, ny = dx / len;
    if (!state.line.inside_is_left) { nx = -nx; ny = -ny; }
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    const tipX = mx + nx * 46, tipY = my + ny * 46;
    const stem = $("arrow-stem");
    stem.setAttribute("x1", mx); stem.setAttribute("y1", my);
    stem.setAttribute("x2", tipX); stem.setAttribute("y2", tipY);
    const px = -ny, py = nx;                       // perpendicular to the arrow
    $("arrow-head").setAttribute(
      "points",
      [
        `${tipX + nx * 10},${tipY + ny * 10}`,
        `${tipX + px * 7},${tipY + py * 7}`,
        `${tipX - px * 7},${tipY - py * 7}`,
      ].join(" ")
    );
  }

  let dragging = null;
  for (const [id, keys] of [["handle-a", ["x1", "y1"]], ["handle-b", ["x2", "y2"]]]) {
    $(id).addEventListener("pointerdown", (e) => {
      dragging = keys;
      $(id).setPointerCapture(e.pointerId);
      e.preventDefault();
    });
  }
  overlay.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = $("preview").getBoundingClientRect();
    const nxp = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const nyp = clamp((e.clientY - rect.top) / rect.height, 0, 1);
    state.line[dragging[0]] = nxp;
    state.line[dragging[1]] = nyp;
    drawLine();
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach((ev) =>
    overlay.addEventListener(ev, () => { dragging = null; })
  );

  $("btn-flip").addEventListener("click", () => {
    state.line.inside_is_left = !state.line.inside_is_left;
    drawLine();
  });
  $("btn-horizontal").addEventListener("click", () => {
    // Drawn right-to-left so the "inside" half-plane is the top of the frame:
    // walking away from the camera reads as entering.
    state.line = { x1: 1, y1: 0.55, x2: 0, y2: 0.55, inside_is_left: true };
    drawLine();
  });
  $("btn-vertical").addEventListener("click", () => {
    state.line = { x1: 0.5, y1: 0, x2: 0.5, y2: 1, inside_is_left: true };
    drawLine();
  });

  /* ---------------------------------------------------------- settings */
  const sliders = [
    ["opt-confidence", "out-confidence", (v) => Number(v).toFixed(2)],
    ["opt-stride", "out-stride", (v) => v],
    ["opt-minhits", "out-minhits", (v) => v],
    ["opt-maxage", "out-maxage", (v) => v],
  ];
  sliders.forEach(([input, out, fmt]) => {
    const el = $(input);
    const sync = () => { $(out).textContent = fmt(el.value); };
    el.addEventListener("input", sync);
    sync();
  });

  /* --------------------------------------------------------------- run */
  $("btn-run").addEventListener("click", startJob);
  $("btn-cancel").addEventListener("click", cancelJob);
  $("btn-again").addEventListener("click", () => {
    fileInput.value = "";
    $("step-upload").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  async function startJob() {
    if (!state.video) return;
    hide($("job-error"));
    hide($("step-results"));
    $("btn-run").disabled = true;
    show($("btn-cancel"));

    const bar = $("job-progress");
    show(bar);
    setBar(bar, 0, "Starting…");

    const payload = {
      video_id: state.video.video_id,
      original_name: state.video.original_name,
      line: state.line,
      detector: $("opt-detector").value,
      confidence: Number($("opt-confidence").value),
      frame_stride: Number($("opt-stride").value),
      min_hits: Number($("opt-minhits").value),
      max_age: Number($("opt-maxage").value),
      write_video: $("opt-video").checked,
    };

    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      state.jobId = data.job_id;
      state.poll = setInterval(pollJob, 700);
      pollJob();
    } catch (err) {
      finishRun();
      hide(bar);
      fail($("job-error"), err.message);
    }
  }

  async function cancelJob() {
    if (!state.jobId) return;
    $("btn-cancel").disabled = true;
    try { await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" }); }
    finally { $("btn-cancel").disabled = false; }
  }

  async function pollJob() {
    if (!state.jobId) return;
    let job;
    try {
      const res = await fetch(`/api/jobs/${state.jobId}`);
      job = await res.json();
    } catch {
      return;                                       // transient; try again next tick
    }

    const bar = $("job-progress");
    const counts = `in ${job.entered} · out ${job.exited}`;
    if (job.status === "queued") {
      setBar(bar, 0, "Queued — waiting for the worker…");
    } else if (job.status === "running" && job.stage === "Loading detector") {
      // A first run downloads model weights, which can take a while and would
      // otherwise look like a frozen 0%.
      setBar(bar, 0.03, "Loading detector — first run may download the model…");
    } else if (job.status === "running") {
      const pct = job.progress ?? 0;
      const eta = job.eta != null ? ` · ~${formatTime(job.eta)} left` : "";
      const of = job.frames_total ? ` / ${job.frames_total}` : "";
      setBar(bar, pct, `${Math.round(pct * 100)}% · frame ${job.frames_done}${of} · ${counts}${eta}`);
    } else {
      clearInterval(state.poll);
      state.poll = null;
      finishRun();
      if (job.status === "failed") {
        hide(bar);
        fail($("job-error"), job.error || "Processing failed.");
        return;
      }
      setBar(bar, 1, job.status === "cancelled"
        ? `Cancelled after ${job.frames_done} frames · ${counts}`
        : `Done in ${formatTime(job.elapsed)} · ${counts}`);
      showResults(job);
    }
  }

  function finishRun() {
    $("btn-run").disabled = false;
    hide($("btn-cancel"));
  }

  /* ----------------------------------------------------------- results */
  function showResults(job) {
    const r = job.result || {};
    $("count-in").textContent = job.entered;
    $("count-out").textContent = job.exited;
    const net = job.occupancy;
    $("count-net").textContent = net > 0 ? `+${net}` : String(net);

    const v = r.video || {};
    $("result-meta").textContent = [
      `detector: ${r.detector ?? "?"}`,
      v.width ? `${v.width}×${v.height} @ ${v.fps} fps` : null,
      `${r.frames_processed ?? job.frames_done} frames in ${formatTime(r.elapsed ?? job.elapsed)}`,
      r.processing_fps ? `${r.processing_fps} fps` : null,
    ].filter(Boolean).join(" · ");

    const video = $("result-video");
    const mp4 = $("btn-mp4");
    if (job.has_video) {
      video.src = `/api/jobs/${job.job_id}/video?t=${Date.now()}`;
      mp4.href = video.src;
      mp4.setAttribute("download", `count-${job.job_id}.mp4`);
      show(video); show(mp4);
    } else {
      video.removeAttribute("src");
      hide(video); hide(mp4);
    }
    $("btn-json").href = `/api/jobs/${job.job_id}/result.json`;

    const list = $("events");
    list.innerHTML = "";
    const events = r.events || [];
    for (const e of events) {
      const li = document.createElement("li");
      const tag = document.createElement("span");
      tag.className = `tag ${e.direction}`;
      tag.textContent = e.direction.toUpperCase();
      const time = document.createElement("span");
      time.className = "t";
      time.textContent = formatTime(e.timestamp);
      const who = document.createElement("span");
      who.textContent = `person #${e.track_id}`;
      li.append(tag, time, who);
      list.append(li);
    }
    toggle($("events"), events.length > 0);
    toggle($("events-title"), events.length > 0);

    show($("step-results"));
    $("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ------------------------------------------------------------ helpers */
  function show(el) { el.classList.remove("hidden"); }
  function hide(el) { el.classList.add("hidden"); }
  function toggle(el, on) { on ? show(el) : hide(el); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function fail(el, msg) { el.textContent = msg; show(el); }
  function setBar(bar, fraction, label) {
    bar.querySelector(".bar-fill").style.width = `${clamp(fraction, 0, 1) * 100}%`;
    bar.querySelector(".bar-label").textContent = label;
  }
  function formatTime(seconds) {
    const s = Number(seconds) || 0;
    if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)}s`;
    return `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`;
  }

  drawLine();
})();
