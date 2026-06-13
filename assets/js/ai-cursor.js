import { magicMouse } from "https://cdn.jsdelivr.net/npm/magicmouse.js";

(function () {
  "use strict";

  function shouldDisableCursor() {
    var reducedMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var coarsePointer = window.matchMedia &&
      window.matchMedia("(pointer: coarse)").matches;
    var smallScreen = window.innerWidth < 768;
    return reducedMotion || coarsePointer || smallScreen;
  }

  function updateCursorGlow(event) {
    document.documentElement.style.setProperty("--cursor-x", event.clientX + "px");
    document.documentElement.style.setProperty("--cursor-y", event.clientY + "px");
    document.body.classList.remove("cursor-idle");
  }

  function addHoverTargets() {
    var selectors = [
      "a",
      "button",
      ".btn",
      ".hero-card",
      ".metric-grid article",
      ".capability-card",
      ".project-card",
      ".timeline > div",
      ".skill-cloud span"
    ];

    document.querySelectorAll(selectors.join(",")).forEach(function (element) {
      element.classList.add("magic-hover");
    });
  }

  function initMagicCursor() {
    if (shouldDisableCursor()) {
      return;
    }

    addHoverTargets();
    document.body.classList.add("magic-cursor-enabled");
    window.addEventListener("pointermove", updateCursorGlow, { passive: true });

    magicMouse({
      outerWidth: 34,
      outerHeight: 34,
      outerStyle: "circle",
      hoverEffect: "pointer-blur",
      hoverItemMove: false,
      defaultCursor: false
    });
  }

  document.addEventListener("DOMContentLoaded", initMagicCursor);
})();
