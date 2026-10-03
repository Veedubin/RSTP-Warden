// warden.js -- shared browser behavior for the rtsp-warden web UI.

// Attach the CSRF token to every htmx request so templates do not need
// per-form hx-headers. The token is rendered into <meta name="csrf-token">.
document.addEventListener("htmx:configRequest", function (evt) {
  var meta = document.querySelector('meta[name="csrf-token"]');
  if (meta && meta.content) {
    evt.detail.headers["X-CSRF-Token"] = meta.content;
  }
});

// htmx 2 does not swap 4xx responses. Routes that answer a bad form post (422)
// or a missing camera or preset (404) send an HTML fragment meant for the page,
// so swap 4xx responses whose body is HTML. JSON errors (CSRF, auth, FastAPI
// validation) are still not swapped.
document.addEventListener("htmx:beforeSwap", function (evt) {
  var xhr = evt.detail.xhr;
  if (!xhr || xhr.status < 400 || xhr.status >= 500) {
    return;
  }
  var contentType = xhr.getResponseHeader("Content-Type") || "";
  if (contentType.indexOf("text/html") === 0) {
    evt.detail.shouldSwap = true;
    evt.detail.isError = false;
  }
});
