// warden.js -- shared browser behavior for the rtsp-warden web UI.

// Attach the CSRF token to every htmx request so templates do not need
// per-form hx-headers. The token is rendered into <meta name="csrf-token">.
document.addEventListener("htmx:configRequest", function (evt) {
  var meta = document.querySelector('meta[name="csrf-token"]');
  if (meta && meta.content) {
    evt.detail.headers["X-CSRF-Token"] = meta.content;
  }
});
