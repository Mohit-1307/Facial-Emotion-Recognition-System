const CLASSES = window.__DEEPFER_CLASSES__ || [];

// Minimal single-stroke line icons (24x24, matching the empty-state glyph
// style already used in the viewfinder) instead of OS emoji -- consistent
// rendering across every platform, and colored via currentColor so each
// one inherits its class's palette color at the call site.

const ICON = {
  angry:
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9.25" stroke="currentColor" stroke-width="1.4"/><path d="M7.2 9.6 9.8 10.8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M16.8 9.6 14.2 10.8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="9.3" cy="12.4" r="0.9" fill="currentColor"/><circle cx="14.7" cy="12.4" r="0.9" fill="currentColor"/><path d="M8.6 17c1-1.1 2.2-1.6 3.4-1.6s2.4.5 3.4 1.6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',

  disgust:
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9.25" stroke="currentColor" stroke-width="1.4"/><path d="M8 10.2h2.4M13.6 10.2H16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M8.4 16.4c1.1-1.6 2.3-2.4 3.6-2.4s2.5.8 3.6 2.4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M9.6 16.9c.9.5 1.7.5 2.4 0" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>',

  fear: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9.25" stroke="currentColor" stroke-width="1.4"/><path d="M7.3 9.3 9.6 10.1" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><path d="M16.7 9.3 14.4 10.1" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><circle cx="9.1" cy="12.1" r="0.9" fill="currentColor"/><circle cx="14.9" cy="12.1" r="0.9" fill="currentColor"/><path d="M10.3 16.6c.5-.7 1.1-1 1.7-1s1.2.3 1.7 1" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',

  happy:
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9.25" stroke="currentColor" stroke-width="1.4"/><circle cx="9.3" cy="11" r="1" fill="currentColor"/><circle cx="14.7" cy="11" r="1" fill="currentColor"/><path d="M7.6 14.4c1.2 1.8 2.8 2.7 4.4 2.7s3.2-.9 4.4-2.7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',

  neutral:
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9.25" stroke="currentColor" stroke-width="1.4"/><circle cx="9.3" cy="11" r="1" fill="currentColor"/><circle cx="14.7" cy="11" r="1" fill="currentColor"/><path d="M8 15.6h8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',

  sad: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9.25" stroke="currentColor" stroke-width="1.4"/><circle cx="9.3" cy="11" r="1" fill="currentColor"/><circle cx="14.7" cy="11" r="1" fill="currentColor"/><path d="M7.6 17c1.2-1.8 2.8-2.7 4.4-2.7s3.2.9 4.4 2.7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M14.6 12.6v2.1" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>',

  surprise:
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9.25" stroke="currentColor" stroke-width="1.4"/><circle cx="9.3" cy="11" r="1" fill="currentColor"/><circle cx="14.7" cy="11" r="1" fill="currentColor"/><ellipse cx="12" cy="16.1" rx="1.6" ry="2" stroke="currentColor" stroke-width="1.3"/></svg>',
};

const COLOR_VAR = {
  angry: "--c-angry",

  disgust: "--c-disgust",

  fear: "--c-fear",

  happy: "--c-happy",

  neutral: "--c-neutral",

  sad: "--c-sad",

  surprise: "--c-surprise",
};

const root = document.documentElement;

const cssColor = (cls) =>
  getComputedStyle(root).getPropertyValue(COLOR_VAR[cls]).trim();

// ---------- elements ----------
const viewfinder = document.getElementById("viewfinder");

const previewImg = document.getElementById("preview-img");

const cameraVideo = document.getElementById("camera-video");

const overlayCanvas = document.getElementById("overlay-canvas");

const emptyState = document.getElementById("empty-state");

const dropVeil = document.getElementById("drop-veil");

const scanline = document.getElementById("scanline");

const fileInput = document.getElementById("file-input");

const cameraStartBtn = document.getElementById("camera-start");

const cameraStopBtn = document.getElementById("camera-stop");

const fpsReadout = document.getElementById("fps-readout");

const primaryEmoji = document.getElementById("primary-emoji");

const primaryLabel = document.getElementById("primary-label");

const primaryConfidence = document.getElementById("primary-confidence");

const faceCountEl = document.getElementById("face-count");

const spectrumEl = document.getElementById("spectrum");

const modeButtons = document.querySelectorAll(".mode-btn");

const healthDot = document.getElementById("health-dot");

const shortcutsBtn = document.getElementById("shortcuts-btn");

const themeToggle = document.getElementById("theme-toggle");

const shortcutsPanel = document.getElementById("shortcuts-panel");

const exportBtn = document.getElementById("export-btn");

const sessionStrip = document.getElementById("session-strip");

const clearLogBtn = document.getElementById("clear-log");

const thresholdWrap = document.getElementById("threshold-wrap");

const thresholdInput = document.getElementById("threshold");

const thresholdValue = document.getElementById("threshold-value");

const traceWrap = document.getElementById("trace-wrap");

const traceCanvas = document.getElementById("trace-canvas");

let mode = "upload";

let cameraStream = null;

let cameraLoopHandle = null;

let lastResult = null;

let threshold = 0;

let traceHistory = [];

const TRACE_MAX_POINTS = 60;

// ---------- build spectrum rows once ----------
function buildSpectrum() {
  spectrumEl.innerHTML = "";

  CLASSES.forEach((cls) => {
    const row = document.createElement("div");

    row.className = "spectrum__row";

    row.id = `row-${cls}`;

    row.innerHTML = `
      <div class="spectrum__label">${cls}</div>
      <div class="spectrum__track"><div class="spectrum__fill" id="fill-${cls}"></div></div>
      <div class="spectrum__value" id="value-${cls}">0%</div>
    `;

    spectrumEl.appendChild(row);

    row.querySelector(`#fill-${cls}`).style.background = cssColor(cls);
  });
}

buildSpectrum();

// ---------- health check ----------
function pingHealth() {
  fetch("/healthz")
    .then((r) => (r.ok ? r.json() : Promise.reject()))
    .then(() => healthDot.classList.add("is-ok"))
    .catch(() => healthDot.classList.add("is-error"));
}

pingHealth();

// ---------- mode switching ----------
modeButtons.forEach((btn) =>
  btn.addEventListener("click", () => setMode(btn.dataset.mode)),
);

function setMode(newMode) {
  mode = newMode;

  modeButtons.forEach((b) => {
    const active = b.dataset.mode === mode;

    b.classList.toggle("is-active", active);

    b.setAttribute("aria-selected", active);
  });

  stopCamera();

  clearOverlay();

  resetReading();

  if (mode === "upload") {
    cameraVideo.hidden = true;

    cameraStartBtn.hidden = true;

    cameraStopBtn.hidden = true;

    fpsReadout.hidden = true;

    thresholdWrap.hidden = true;

    previewImg.hidden = !previewImg.src;

    emptyState.hidden = !!previewImg.src;

    viewfinder.onclick = () => fileInput.click();
  } else {
    previewImg.hidden = true;

    cameraStartBtn.hidden = false;

    thresholdWrap.hidden = false;

    emptyState.hidden = true;

    viewfinder.onclick = null;
  }
}

// ---------- upload flow ----------
viewfinder.onclick = () => fileInput.click();

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];

  if (file) loadImageFile(file);
});

viewfinder.addEventListener("dragover", (e) => {
  e.preventDefault();

  if (mode === "upload") dropVeil.hidden = false;
});

viewfinder.addEventListener("dragleave", () => {
  dropVeil.hidden = true;
});

viewfinder.addEventListener("drop", (e) => {
  e.preventDefault();

  dropVeil.hidden = true;

  if (mode !== "upload") return;

  const file = e.dataTransfer.files[0];

  if (file) loadImageFile(file);
});

function loadImageFile(file) {
  const reader = new FileReader();

  reader.onload = (e) => {
    previewImg.src = e.target.result;

    previewImg.hidden = false;

    emptyState.hidden = true;

    previewImg.onload = () => {
      sizeOverlayTo(previewImg);

      scanline.hidden = false;

      const form = new FormData();

      form.append("image", file);

      fetch("/predict", { method: "POST", body: form })
        .then((r) => r.json())
        .then((data) => {
          scanline.hidden = true;

          handlePrediction(data, previewImg);
        })
        .catch(() => {
          scanline.hidden = true;

          resetReading("Error reaching server");
        });
    };
  };

  reader.readAsDataURL(file);
}

// ---------- camera flow ----------
cameraStartBtn.addEventListener("click", startCamera);

cameraStopBtn.addEventListener("click", stopCamera);

function startCamera() {
  navigator.mediaDevices
    .getUserMedia({ video: { width: 640, height: 480 } })
    .then((stream) => {
      cameraStream = stream;

      cameraVideo.srcObject = stream;

      cameraVideo.hidden = false;

      cameraStartBtn.hidden = true;

      cameraStopBtn.hidden = false;

      fpsReadout.hidden = false;

      viewfinder.classList.add("is-active");

      cameraVideo.onloadedmetadata = () => {
        sizeOverlayTo(cameraVideo);

        cameraLoop();
      };
    })
    .catch(() => {
      fpsReadout.hidden = false;

      fpsReadout.textContent = "Camera unavailable / permission denied";
    });
}

function stopCamera() {
  if (cameraLoopHandle) clearTimeout(cameraLoopHandle);

  cameraLoopHandle = null;

  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());

    cameraStream = null;
  }

  cameraVideo.hidden = true;

  cameraStartBtn.hidden = mode !== "camera";

  cameraStopBtn.hidden = true;

  viewfinder.classList.remove("is-active");
}

function cameraLoop() {
  const captureCanvas = document.createElement("canvas");

  captureCanvas.width = cameraVideo.videoWidth;

  captureCanvas.height = cameraVideo.videoHeight;

  const ctx = captureCanvas.getContext("2d");

  const tick = () => {
    if (!cameraStream) return;

    const t0 = performance.now();

    ctx.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);

    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);

    fetch("/predict_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    })
      .then((r) => r.json())
      .then((data) => {
        handlePrediction(data, cameraVideo);

        const roundtrip = (performance.now() - t0).toFixed(0);

        fpsReadout.textContent = `${(1000 / (performance.now() - t0)).toFixed(1)} FPS · ${roundtrip} ms`;
      })
      .catch(() => {})
      .finally(() => {
        cameraLoopHandle = setTimeout(tick, 350); // throttled -- see README "Real-time performance notes"
      });
  };

  tick();
}

// ---------- threshold filter ----------
thresholdInput.addEventListener("input", () => {
  threshold = Number(thresholdInput.value);

  thresholdValue.textContent = `${threshold}%`;

  if (lastResult) applyThresholdVisual(lastResult);
});

function applyThresholdVisual(top) {
  CLASSES.forEach((cls) => {
    const row = document.getElementById(`row-${cls}`);

    const p = (top.probabilities[cls] || 0) * 100;

    if (row) row.classList.toggle("is-below-threshold", p < threshold);
  });
}

// ---------- confidence trace ----------
function pushTrace(confidence) {
  traceHistory.push(confidence);

  if (traceHistory.length > TRACE_MAX_POINTS) traceHistory.shift();

  drawTrace();
}

function drawTrace() {
  const ctx = traceCanvas.getContext("2d");

  const w = traceCanvas.width;

  const h = traceCanvas.height;

  ctx.clearRect(0, 0, w, h);

  if (traceHistory.length === 0) return;

  const signalColor = getComputedStyle(root)
    .getPropertyValue("--signal")
    .trim();

  if (traceHistory.length === 1) {
    const y = h - traceHistory[0] * h * 0.9 - h * 0.05;

    ctx.fillStyle = signalColor;

    ctx.beginPath();

    ctx.arc(4, y, 3, 0, Math.PI * 2);

    ctx.fill();

    return;
  }

  ctx.strokeStyle = signalColor;

  ctx.lineWidth = 1.6;

  ctx.beginPath();

  traceHistory.forEach((v, i) => {
    const x = (i / (TRACE_MAX_POINTS - 1)) * w;

    const y = h - v * h * 0.9 - h * 0.05;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();

  // faint fill under the trace
  ctx.lineTo(((traceHistory.length - 1) / (TRACE_MAX_POINTS - 1)) * w, h);

  ctx.lineTo(0, h);

  ctx.closePath();

  ctx.fillStyle = getComputedStyle(root)
    .getPropertyValue("--signal-soft")
    .trim();

  ctx.globalAlpha = 0.35;

  ctx.fill();

  ctx.globalAlpha = 1;
}

// ---------- session log ----------
function logReading(top) {
  const emptyMsg = sessionStrip.querySelector(".session-log__empty");

  if (emptyMsg) emptyMsg.remove();

  const chip = document.createElement("div");

  chip.className = "session-chip";

  chip.innerHTML = ICON[top.label] || "";

  chip.title = `${top.label} · ${(top.confidence * 100).toFixed(0)}%`;

  chip.dataset.pct = `${(top.confidence * 100).toFixed(0)}`;

  chip.style.borderColor = cssColor(top.label);

  chip.style.color = cssColor(top.label);

  sessionStrip.appendChild(chip);

  sessionStrip.scrollLeft = sessionStrip.scrollWidth;

  clearLogBtn.hidden = false;

  while (sessionStrip.children.length > 40) {
    sessionStrip.removeChild(sessionStrip.firstChild);
  }
}

clearLogBtn.addEventListener("click", () => {
  sessionStrip.innerHTML =
    '<div class="session-log__empty">Readings will appear here</div>';

  clearLogBtn.hidden = true;
});

// ---------- export ----------
exportBtn.addEventListener("click", exportReading);

function exportReading() {
  if (!lastResult) return;

  const blob = new Blob([JSON.stringify(lastResult, null, 2)], {
    type: "application/json",
  });

  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");

  a.href = url;

  a.download = `deepfer-reading-${Date.now()}.json`;

  a.click();

  URL.revokeObjectURL(url);
}

// ---------- shared rendering ----------
function sizeOverlayTo(mediaEl) {
  const rect = mediaEl.getBoundingClientRect();

  overlayCanvas.width = rect.width;

  overlayCanvas.height = rect.height;

  overlayCanvas.style.width = rect.width + "px";

  overlayCanvas.style.height = rect.height + "px";
}

function clearOverlay() {
  const ctx = overlayCanvas.getContext("2d");

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function handlePrediction(data, mediaEl) {
  clearOverlay();

  if (!data || !data.faces || data.faces.length === 0) {
    faceCountEl.textContent = "No face detected";

    resetReading();

    return;
  }

  faceCountEl.textContent = `${data.faces.length} face${data.faces.length > 1 ? "s" : ""} detected · ${data.inference_ms} ms inference`;

  const naturalW = mediaEl.videoWidth || mediaEl.naturalWidth;

  const naturalH = mediaEl.videoHeight || mediaEl.naturalHeight;

  const scaleX = overlayCanvas.width / naturalW;

  const scaleY = overlayCanvas.height / naturalH;

  const ctx = overlayCanvas.getContext("2d");

  data.faces.forEach((face) => {
    const color = cssColor(face.label);

    ctx.strokeStyle = color;

    ctx.lineWidth = 2;

    ctx.strokeRect(
      face.box.x * scaleX,
      face.box.y * scaleY,
      face.box.w * scaleX,
      face.box.h * scaleY,
    );

    ctx.font = "500 13px 'JetBrains Mono', monospace";

    ctx.fillStyle = color;

    ctx.fillText(
      `${face.label} ${(face.confidence * 100).toFixed(0)}%`,
      face.box.x * scaleX,
      face.box.y * scaleY - 6,
    );
  });

  const top = data.faces.reduce((a, b) =>
    a.confidence > b.confidence ? a : b,
  );

  lastResult = { ...data, timestamp: new Date().toISOString() };

  primaryEmoji.innerHTML = ICON[top.label] || "";

  primaryEmoji.style.color = cssColor(top.label);

  primaryEmoji.classList.remove("is-updated");

  void primaryEmoji.offsetWidth; // restart animation

  primaryEmoji.classList.add("is-updated");

  primaryLabel.textContent = top.label;

  primaryLabel.style.color = cssColor(top.label);

  primaryConfidence.textContent = `${(top.confidence * 100).toFixed(1)}% confidence`;

  CLASSES.forEach((cls) => {
    const p = (top.probabilities[cls] || 0) * 100;

    const fill = document.getElementById(`fill-${cls}`);

    const value = document.getElementById(`value-${cls}`);

    if (fill) fill.style.width = `${p}%`;

    if (value) value.textContent = `${p.toFixed(0)}%`;
  });

  applyThresholdVisual(top);

  exportBtn.hidden = false;

  traceWrap.hidden = false;

  pushTrace(top.confidence);

  logReading(top);
}

function resetReading(message) {
  primaryEmoji.textContent = "—";

  primaryEmoji.style.color = "";

  primaryEmoji.classList.remove("is-updated");

  primaryLabel.textContent = message || "Awaiting input";

  primaryLabel.style.color = "";

  primaryConfidence.innerHTML = "&nbsp;";

  faceCountEl.textContent = "";

  CLASSES.forEach((cls) => {
    const fill = document.getElementById(`fill-${cls}`);

    const value = document.getElementById(`value-${cls}`);

    const row = document.getElementById(`row-${cls}`);

    if (fill) fill.style.width = "0%";

    if (value) value.textContent = "0%";

    if (row) row.classList.remove("is-below-threshold");
  });
}

resetReading();

// ---------- theme ----------
const THEME_KEY = "deepfer-theme";

function applyTheme(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }

  localStorage.setItem(THEME_KEY, theme);
}

themeToggle.addEventListener("click", () => {
  const current =
    document.documentElement.getAttribute("data-theme") === "light"
      ? "light"
      : "dark";

  applyTheme(current === "light" ? "dark" : "light");
});

// ---------- keyboard shortcuts ----------
shortcutsBtn.addEventListener("click", toggleShortcuts);

function toggleShortcuts() {
  shortcutsPanel.hidden = !shortcutsPanel.hidden;
}

shortcutsPanel.addEventListener("click", (e) => {
  if (e.target === shortcutsPanel) shortcutsPanel.hidden = true;
});

document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();

  if (tag === "input" || tag === "textarea") return;

  switch (e.key.toLowerCase()) {
    case "u":
      setMode("upload");

      break;

    case "c":
      setMode("camera");

      break;

    case " ":
      e.preventDefault();

      if (mode === "camera") {
        cameraStream ? stopCamera() : startCamera();
      }

      break;

    case "escape":
      if (!shortcutsPanel.hidden) {
        shortcutsPanel.hidden = true;
      } else {
        clearOverlay();

        resetReading();
      }

      break;

    case "e":
      exportReading();

      break;

    case "t":
      themeToggle.click();

      break;

    case "?":
      toggleShortcuts();

      break;

    default:
      break;
  }
});
