(function () {
  var root = document.documentElement;
  var idleTimer;
  var lastX = window.innerWidth / 2;
  var lastY = window.innerHeight / 2;
  var trail = document.createElement("span");
  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduceMotion || !window.PointerEvent) {
    return;
  }

  trail.className = "cursor-trail";
  trail.setAttribute("aria-hidden", "true");
  document.body.appendChild(trail);

  function setCursorPosition(event) {
    var dx = event.clientX - lastX;
    var dy = event.clientY - lastY;
    var angle = Math.atan2(dy, dx) * 180 / Math.PI;
    var distance = Math.min(Math.sqrt(dx * dx + dy * dy), 72);
    var offsetX = distance ? dx / distance * 18 : 0;
    var offsetY = distance ? dy / distance * 18 : 0;

    root.style.setProperty("--cursor-x", event.clientX + "px");
    root.style.setProperty("--cursor-y", event.clientY + "px");
    root.style.setProperty("--trail-x", (event.clientX - offsetX) + "px");
    root.style.setProperty("--trail-y", (event.clientY - offsetY) + "px");
    root.style.setProperty("--trail-angle", angle + "deg");
    root.style.setProperty("--trail-opacity", distance > 2 ? "1" : "0.45");
    document.body.classList.remove("cursor-idle");

    window.clearTimeout(idleTimer);
    idleTimer = window.setTimeout(function () {
      document.body.classList.add("cursor-idle");
      root.style.setProperty("--trail-opacity", "0");
    }, 1200);

    lastX = event.clientX;
    lastY = event.clientY;
  }

  window.addEventListener("pointermove", setCursorPosition, { passive: true });
})();
