"""The camera page: a browser page that shows a camera's live view with the grid found in it and
takes the clock's photos.

The page opens the camera of the device it runs on (a USB camera on the computer, later a phone)
and asks the server for the next photo all the time (a long poll on /camera/next). When a player
presses the clock, the server asks for a photo (CameraPage.request); the page takes a full
resolution frame and sends it (/camera/photo), and the clock's Scorer gets it from
CameraPage.frame. Every second or so the page also sends a small frame to /camera/grid and draws
the 15x15 grid the corner model finds in it over the live view, to aim the camera.

Browsers give a page the camera only on http://localhost or over https: a phone opens the page
over https on the clock's second port (scrabble.clock).
"""

import threading
import time
from collections import deque

import cv2
import numpy as np

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scrabble camera</title>
<style>
  :root { --bg: #111; --panel: #222; --text: #eee; --dim: #888; --grid: #4caf50; --warn: #ffb300; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
               font-family: system-ui, sans-serif; }
  #bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 8px 16px; }
  #bar select, #bar button { background: var(--panel); color: var(--text); border: 1px solid #444;
                             border-radius: 8px; padding: 6px 10px; font: inherit; }
  #status { color: var(--dim); margin-left: auto; }
  #view { display: flex; justify-content: center; padding: 0 16px; }
  #frame { position: relative; display: inline-block; line-height: 0; }
  video { display: block; max-width: 100%; max-height: calc(100vh - 64px); background: #000; }
  canvas#grid { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  #flash { position: absolute; inset: 0; background: #fff; opacity: 0; transition: opacity .4s; }
  #flash.on { opacity: .7; transition: none; }
</style>
</head>
<body>
<div id="bar">
  <select id="device" title="Camera"></select>
  <button id="toggle" title="G">Grid: on</button>
  <span id="status">Starting the camera…</span>
</div>
<div id="view"><div id="frame">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="grid"></canvas>
  <div id="flash"></div>
</div></div>
<script>
const video = document.getElementById("video"), overlay = document.getElementById("grid");
const select = document.getElementById("device"), toggle = document.getElementById("toggle");
let showGrid = true, lines = null, conf = null, photos = 0, info = "";

function status(text) { info = text; document.getElementById("status").textContent = text; }

function remembered() { try { return localStorage.getItem("camera"); } catch (e) { return null; } }
function remember(id) { try { localStorage.setItem("camera", id); } catch (e) {} }

async function start(deviceId) {
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  // Ask for a huge size: the browser gives the camera's largest one. A phone starts with its back camera.
  const size = {width: {ideal: 4096}, height: {ideal: 2160}};
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia(
      {video: deviceId ? {deviceId: {exact: deviceId}, ...size} : {facingMode: {ideal: "environment"}, ...size},
       audio: false});
  } catch (e) {
    if (deviceId) return start(null);     // the remembered camera is gone
    status("Can't open the camera: " + e.message); return;
  }
  const id = video.srcObject.getVideoTracks()[0].getSettings().deviceId;
  if (id) remember(id);
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
  select.innerHTML = "";
  cameras.forEach((d, i) => select.add(new Option(d.label || "Camera " + (i + 1), d.deviceId, false, d.deviceId === id)));
  status("Waiting for the clock");
}

// Keep a phone's screen on during the game; the browser drops the lock when the page is hidden.
async function stayAwake() {
  try { if (navigator.wakeLock && document.visibilityState === "visible") await navigator.wakeLock.request("screen"); }
  catch (e) {}
}
document.addEventListener("visibilitychange", stayAwake);

function frame(maxSide) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w) return null;
  const s = Math.min(1, maxSide / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * s); canvas.height = Math.round(h * s);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function jpeg(canvas, quality) {
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", quality));
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function photoLoop() {
  let after = 0;
  for (;;) {
    try {
      const r = await fetch("camera/next?after=" + after);
      const shot = (await r.json()).shot;
      if (shot === null) continue;
      after = shot;
      const canvas = frame(100000);
      if (!canvas) { status("No picture yet: the clock's photo was not taken"); continue; }
      const flash = document.getElementById("flash");
      flash.classList.add("on"); setTimeout(() => flash.classList.remove("on"), 50);
      await fetch("camera/photo?shot=" + shot, {method: "POST", headers: {"Content-Type": "image/jpeg"},
                                                body: await jpeg(canvas, 0.95)});
      photos++;
      status("Photos taken: " + photos + "  ·  waiting for the clock");
    } catch (e) {
      status("The server did not answer");
      await sleep(1000);
    }
  }
}

async function gridLoop() {
  for (;;) {
    const canvas = showGrid ? frame(1280) : null;
    if (canvas) {
      try {
        const r = await fetch("camera/grid", {method: "POST", headers: {"Content-Type": "image/jpeg"},
                                              body: await jpeg(canvas, 0.85)});
        ({lines, conf} = r.status === 200 ? await r.json() : {lines: null, conf: null});
      } catch (e) { lines = null; }
      draw();
    }
    await sleep(showGrid ? 300 : 1000);
  }
}

function draw() {
  const w = overlay.clientWidth, h = overlay.clientHeight;
  overlay.width = w * devicePixelRatio; overlay.height = h * devicePixelRatio;
  const ctx = overlay.getContext("2d");
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!showGrid || !lines) return;
  // Weak corners usually mean the board is not (all) in view.
  const sure = Math.min(...conf) >= 0.3;
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue(sure ? "--grid" : "--warn");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  lines.forEach(([[x1, y1], [x2, y2]]) => { ctx.moveTo(x1 * w, y1 * h); ctx.lineTo(x2 * w, y2 * h); });
  ctx.stroke();
}

function setGrid(on) {
  showGrid = on;
  toggle.textContent = "Grid: " + (on ? "on" : "off");
  draw();
}

toggle.onclick = () => setGrid(!showGrid);
select.onchange = () => start(select.value);
document.addEventListener("keydown", e => { if (e.key === "g" || e.key === "G") setGrid(!showGrid); });
window.addEventListener("resize", draw);
if (!navigator.mediaDevices) status("No camera: open this page on http://localhost or over https");
else start(remembered()).then(() => { stayAwake(); photoLoop(); gridLoop(); });
</script>
</body>
</html>
"""


class CameraPage:
    """The camera page as the clock's camera: `request` asks the page for a photo, `frame` waits for
    it. Photos are asked for and taken in order, one per turn."""

    def __init__(self, timeout=10.0, seen=30.0):
        self.timeout, self.seen = timeout, seen      # s to wait for a photo; s a page counts as open
        self.cond = threading.Condition()
        self.shots = deque()                         # [shot id, JPEG bytes or None], oldest first
        self.next_id = 1
        self.polls, self.last_poll = 0, None         # the page's long polls now open, and the last one's end

    def request(self):
        """Ask the page for a photo of the board as it is now."""
        with self.cond:
            self.shots.append([self.next_id, None])
            self.next_id += 1
            self.cond.notify_all()

    def is_open(self):
        with self.cond:
            return self.polls > 0 or (self.last_poll is not None and time.monotonic() - self.last_poll < self.seen)

    def next_shot(self, after, wait=20.0):
        """For the page: the oldest asked-for photo after shot `after` still to be taken, waiting up
        to `wait` s for one; None if there is none."""
        def pending():
            return next((s[0] for s in self.shots if s[0] > after and s[1] is None), None)
        with self.cond:
            self.polls += 1
            try:
                self.cond.wait_for(pending, wait)
                return pending()
            finally:
                self.polls -= 1
                self.last_poll = time.monotonic()

    def deliver(self, shot, data):
        """For the page: the photo for `shot`, as JPEG bytes. False if nobody waits for it any more."""
        with self.cond:
            for s in self.shots:
                if s[0] == shot and s[1] is None:
                    s[1] = data
                    self.cond.notify_all()
                    return True
            return False

    def frame(self):
        """The oldest asked-for photo, as a BGR image; raises ValueError if it doesn't come."""
        with self.cond:
            if not self.shots:
                raise ValueError("no photo was asked of the camera page")
            shot = self.shots[0]
            if not self.is_open():                   # the condition's lock is re-entrant
                self.shots.popleft()
                raise ValueError("the camera page is not open")
            self.cond.wait_for(lambda: shot[1] is not None, self.timeout)
            self.shots.popleft()
        if shot[1] is None:
            raise ValueError(f"the camera page sent no photo in {self.timeout:g} s")
        image = cv2.imdecode(np.frombuffer(shot[1], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("the camera page sent a broken photo")
        return image


def grid_finder(corner_model, fit=True):
    """A function JPEG bytes -> {"lines": the grid lines with coordinates as fractions of the
    image's width and height, "conf": the 4 corners' confidence}, with the model loaded once."""
    from scrabble.corners import grid_lines, predict_corners
    lock = threading.Lock()

    def find(data):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("not an image")
        h, w = image.shape[:2]
        with lock:
            corners, conf = predict_corners(corner_model, image, fit=fit)
        lines = grid_lines(corners) / [w, h]
        return {"lines": np.round(lines, 4).tolist(), "conf": [round(float(c), 3) for c in conf]}

    return find
