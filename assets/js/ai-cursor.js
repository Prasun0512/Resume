(function () {
  var root = document.documentElement;
  var idleTimer;
  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduceMotion || !window.PointerEvent) {
    return;
  }

  function setCursorPosition(event) {
    root.style.setProperty("--cursor-x", event.clientX + "px");
    root.style.setProperty("--cursor-y", event.clientY + "px");
    document.body.classList.remove("cursor-idle");

    window.clearTimeout(idleTimer);
    idleTimer = window.setTimeout(function () {
      document.body.classList.add("cursor-idle");
    }, 1200);
  }

  window.addEventListener("pointermove", setCursorPosition, { passive: true });
})();
