(function () {
  "use strict";

  var DEFAULTS = {
    color: "#2ce6be",
    trailLength: 34,
    dotSize: 3.6,
    lineWidth: 2.2,
    opacity: 0.72,
    fadeSpeed: 0.035
  };

  function hexToRgb(hex) {
    var value = hex.replace("#", "").trim();
    if (value.length === 3) {
      value = value.split("").map(function (char) {
        return char + char;
      }).join("");
    }
    var parsed = parseInt(value, 16);
    return {
      r: (parsed >> 16) & 255,
      g: (parsed >> 8) & 255,
      b: parsed & 255
    };
  }

  function shouldDisableTrail() {
    var reducedMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var coarsePointer = window.matchMedia &&
      window.matchMedia("(pointer: coarse)").matches;
    var smallScreen = window.innerWidth < 768;
    return reducedMotion || coarsePointer || smallScreen;
  }

  function CursorTrail(options) {
    this.options = Object.assign({}, DEFAULTS, options || {});
    this.rgb = hexToRgb(this.options.color);
    this.canvas = null;
    this.ctx = null;
    this.points = [];
    this.animationFrame = 0;
    this.running = false;
    this.hasPointer = false;
    this.targetX = window.innerWidth / 2;
    this.targetY = window.innerHeight / 2;
    this.dotX = this.targetX;
    this.dotY = this.targetY;
    this.dotOpacity = 0;

    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handleResize = this.handleResize.bind(this);
    this.animate = this.animate.bind(this);
  }

  CursorTrail.prototype.init = function () {
    if (shouldDisableTrail() || this.running) {
      return;
    }

    this.canvas = document.createElement("canvas");
    this.canvas.className = "cursor-trail-canvas";
    this.canvas.setAttribute("aria-hidden", "true");
    this.ctx = this.canvas.getContext("2d");
    document.body.appendChild(this.canvas);

    this.handleResize();
    window.addEventListener("pointermove", this.handlePointerMove, { passive: true });
    window.addEventListener("resize", this.handleResize, { passive: true });
    this.running = true;
    this.animate();
  };

  CursorTrail.prototype.handlePointerMove = function (event) {
    this.hasPointer = true;
    this.targetX = event.clientX;
    this.targetY = event.clientY;
    this.dotOpacity = this.options.opacity;

    document.documentElement.style.setProperty("--cursor-x", event.clientX + "px");
    document.documentElement.style.setProperty("--cursor-y", event.clientY + "px");
    document.body.classList.remove("cursor-idle");
  };

  CursorTrail.prototype.handleResize = function () {
    var ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.floor(window.innerWidth * ratio);
    this.canvas.height = Math.floor(window.innerHeight * ratio);
    this.canvas.style.width = window.innerWidth + "px";
    this.canvas.style.height = window.innerHeight + "px";
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  };

  CursorTrail.prototype.addPoint = function () {
    if (!this.hasPointer || this.dotOpacity <= 0.01) {
      return;
    }

    this.points.push({
      x: this.dotX,
      y: this.dotY,
      life: this.options.opacity
    });

    while (this.points.length > this.options.trailLength) {
      this.points.shift();
    }
  };

  CursorTrail.prototype.drawTrail = function () {
    var ctx = this.ctx;
    var rgb = this.rgb;
    var lineWidth = this.options.lineWidth;

    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

    for (var i = 1; i < this.points.length; i += 1) {
      var previous = this.points[i - 1];
      var current = this.points[i];
      var alpha = Math.max(current.life, 0);
      var width = Math.max(lineWidth * alpha, 0.4);

      ctx.beginPath();
      ctx.moveTo(previous.x, previous.y);
      ctx.lineTo(current.x, current.y);
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", " + alpha + ")";
      ctx.shadowColor = "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", " + (alpha * 0.9) + ")";
      ctx.shadowBlur = 14;
      ctx.stroke();
    }

    if (this.dotOpacity > 0.01) {
      var radius = this.options.dotSize;
      var gradient = ctx.createRadialGradient(this.dotX, this.dotY, 0, this.dotX, this.dotY, radius * 5);
      gradient.addColorStop(0, "rgba(255, 255, 255, " + this.dotOpacity + ")");
      gradient.addColorStop(0.24, "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", " + this.dotOpacity + ")");
      gradient.addColorStop(1, "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", 0)");

      ctx.beginPath();
      ctx.fillStyle = gradient;
      ctx.shadowBlur = 0;
      ctx.arc(this.dotX, this.dotY, radius * 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.fillStyle = "rgba(255, 255, 255, " + Math.min(this.dotOpacity + 0.08, 1) + ")";
      ctx.arc(this.dotX, this.dotY, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  CursorTrail.prototype.animate = function () {
    var ease = 0.22;
    var fadeSpeed = this.options.fadeSpeed;

    this.dotX += (this.targetX - this.dotX) * ease;
    this.dotY += (this.targetY - this.dotY) * ease;
    this.addPoint();

    this.points = this.points
      .map(function (point) {
        return {
          x: point.x,
          y: point.y,
          life: point.life - fadeSpeed
        };
      })
      .filter(function (point) {
        return point.life > 0;
      });

    if (Math.abs(this.targetX - this.dotX) < 0.2 && Math.abs(this.targetY - this.dotY) < 0.2) {
      this.dotOpacity = Math.max(this.dotOpacity - fadeSpeed, 0);
      if (this.dotOpacity === 0) {
        document.body.classList.add("cursor-idle");
      }
    }

    this.drawTrail();
    this.animationFrame = window.requestAnimationFrame(this.animate);
  };

  CursorTrail.prototype.destroy = function () {
    window.removeEventListener("pointermove", this.handlePointerMove);
    window.removeEventListener("resize", this.handleResize);
    window.cancelAnimationFrame(this.animationFrame);
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
    this.points = [];
    this.running = false;
  };

  window.CursorTrail = CursorTrail;

  document.addEventListener("DOMContentLoaded", function () {
    var config = window.portfolioCursorTrail || {};
    var cursorTrail = new CursorTrail(config);
    cursorTrail.init();
    window.portfolioCursorTrailInstance = cursorTrail;
  });
})();
