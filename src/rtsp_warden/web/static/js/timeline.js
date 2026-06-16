/**
 * Timeline scrubber for the recording detail page.
 *
 * Renders a canvas-based timeline showing recording segments as colored
 * bars and detector events as vertical tick marks. Clicking a position
 * on the timeline loads the HLS playlist starting at that time.
 *
 * Auto-initialises on DOMContentLoaded by looking for
 *   <div id="timeline-mount" data-recording-id="..." data-camera-name="..."
 *        data-stream="..." data-start-ts="...">
 */

/* eslint-disable no-unused-vars */
/* global Hls */

class Timeline {
  /**
   * @param {HTMLElement} mountEl - The #timeline-mount element.
   */
  constructor(mountEl) {
    this.mount = mountEl;
    this.recordingId = parseInt(mountEl.dataset.recordingId, 10);
    this.cameraName = mountEl.dataset.cameraName;
    this.stream = mountEl.dataset.stream;
    this.startTs = parseFloat(mountEl.dataset.startTs) || 0;

    this.canvas = mountEl.querySelector("#timeline-canvas");
    this.tooltip = mountEl.querySelector("#timeline-tooltip");

    /** @type {{start: number, end: number, path: string, size: number}[]} */
    this.segments = [];
    /** @type {{id: number, type: string, severity: string, ts: number}[]} */
    this.events = [];
    /** @type {number} */
    this.timelineStart = 0;
    /** @type {number} */
    this.timelineEnd = 0;

    this._onCanvasClick = this._onCanvasClick.bind(this);
    this._onCanvasMove = this._onCanvasMove.bind(this);
    this._onCanvasLeave = this._onCanvasLeave.bind(this);
  }

  /** Fetch timeline data from the API and render. */
  async init() {
    this.canvas.addEventListener("click", this._onCanvasClick);
    this.canvas.addEventListener("mousemove", this._onCanvasMove);
    this.canvas.addEventListener("mouseleave", this._onCanvasLeave);

    try {
      const resp = await fetch("/api/recordings/" + this.recordingId + "/timeline");
      if (!resp.ok) {
        console.error("Timeline API returned", resp.status);
        return;
      }
      const data = await resp.json();
      this.segments = data.segments || [];
      this.events = data.events || [];
      this.timelineStart = data.start_ts;
      this.timelineEnd = data.end_ts;
    } catch (err) {
      console.error("Failed to load timeline data:", err);
      return;
    }

    this._resize();
    this._draw();
  }

  /** Resize the canvas to match its CSS layout size at 2x DPR. */
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this._ctx().scale(dpr, dpr);
  }

  /** @returns {CanvasRenderingContext2D} */
  _ctx() {
    return this.canvas.getContext("2d");
  }

  /** Render the full timeline onto the canvas. */
  _draw() {
    const ctx = this._ctx();
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;

    // Clear.
    ctx.clearRect(0, 0, w, h);

    const duration = this.timelineEnd - this.timelineStart;
    if (duration <= 0) {
      return;
    }

    const scaleX = (ts) => ((ts - this.timelineStart) / duration) * w;

    // Background.
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, w, h);

    // Segments.
    for (const seg of this.segments) {
      const x1 = scaleX(seg.start);
      const x2 = scaleX(seg.end);
      ctx.fillStyle = "#4caf50";
      ctx.fillRect(x1, 8, Math.max(x2 - x1, 1), h - 16);
    }

    // Events.
    const severityColors = {
      info: "#2196f3",
      warn: "#ff9800",
      error: "#f44336",
    };
    for (const evt of this.events) {
      const x = scaleX(evt.ts);
      ctx.strokeStyle = severityColors[evt.severity] || "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 2);
      ctx.lineTo(x, h - 2);
      ctx.stroke();
    }

    // Border.
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, w, h);
  }

  /**
   * Convert a canvas click event to a unix timestamp.
   * @param {MouseEvent} e
   * @returns {number}
   */
  _clickToTimestamp(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const fraction = x / rect.width;
    return this.timelineStart + fraction * (this.timelineEnd - this.timelineStart);
  }

  /**
   * Load the HLS playlist at the clicked time.
   * @param {MouseEvent} e
   */
  _onCanvasClick(e) {
    const clickTs = this._clickToTimestamp(e);
    const url =
      "/htl/" +
      encodeURIComponent(this.cameraName) +
      "/" +
      encodeURIComponent(this.stream) +
      "/" +
      clickTs.toFixed(3) +
      "/" +
      this.timelineEnd.toFixed(3) +
      ".m3u8";

    const video = document.getElementById("player");
    if (!video) {
      return;
    }

    if (typeof Hls !== "undefined" && Hls.isSupported()) {
      // Reuse or create an HLS instance stored on the video element.
      if (!video._hlsInstance) {
        const hls = new Hls();
        hls.attachMedia(video);
        video._hlsInstance = hls;
      }
      video._hlsInstance.loadSource(url);
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      // Native HLS (Safari).
      video.src = url;
      video.load();
    }
  }

  /**
   * Show a tooltip on hover.
   * @param {MouseEvent} e
   */
  _onCanvasMove(e) {
    const clickTs = this._clickToTimestamp(e);
    const duration = this.timelineEnd - this.timelineStart;
    if (duration <= 0) {
      return;
    }

    const fraction = (clickTs - this.timelineStart) / duration;
    const rect = this.canvas.getBoundingClientRect();
    const mountRect = this.mount.getBoundingClientRect();

    // Find the segment under the cursor (if any).
    let segLabel = "No segment";
    for (const seg of this.segments) {
      if (clickTs >= seg.start && clickTs <= seg.end) {
        segLabel = seg.path;
        break;
      }
    }

    // Count events at this time.
    const nearbyEvents = this.events.filter(
      (ev) => Math.abs(ev.ts - clickTs) < duration * 0.02
    );

    const dateStr = new Date(clickTs * 1000).toISOString().replace("T", " ").slice(0, 19);
    let tip = "Time: " + dateStr + " | Segment: " + segLabel;
    if (nearbyEvents.length > 0) {
      tip += " | Events: " + nearbyEvents.length;
    }

    this.tooltip.textContent = tip;
    this.tooltip.style.display = "block";
    this.tooltip.style.left = e.clientX - mountRect.left + 12 + "px";
    this.tooltip.style.top = e.clientY - mountRect.top - 28 + "px";
  }

  /** Hide the tooltip when the cursor leaves the canvas. */
  _onCanvasLeave() {
    this.tooltip.style.display = "none";
  }
}

// Auto-initialise on DOMContentLoaded.
document.addEventListener("DOMContentLoaded", () => {
  const mount = document.getElementById("timeline-mount");
  if (mount) {
    const tl = new Timeline(mount);
    tl.init();
  }
});