(function () {
  const videoEl = document.getElementById("camera");
  const frameCanvas = document.getElementById("frame-canvas");
  const overlayCanvas = document.getElementById("overlay-canvas");
  const statusEl = document.getElementById("verification-status");
  const progressEl = document.getElementById("verification-progress");
  const stepsEl = document.getElementById("verification-steps");
  const promptEl = document.getElementById("verification-prompt");
  const errorEl = document.getElementById("verification-error");
  const successEl = document.getElementById("verification-success");
  const startBtn = document.getElementById("start-btn");
  const cancelBtn = document.getElementById("cancel-btn");

  if (!videoEl || !frameCanvas || !overlayCanvas || !statusEl || !startBtn || !cancelBtn) {
    return;
  }

  const frameContext = frameCanvas.getContext("2d");
  const overlayContext = overlayCanvas.getContext("2d");

  const STATE = {
    fps: 5,
    quality: 0.7,
    stream: null,
    socket: null,
    sendIntervalId: null,
    reconnectAttempt: 0,
    maxReconnectAttempts: 5,
    isVerifying: false,
  };

  function setProgress(step, text) {
    if (progressEl) {
      progressEl.dataset.step = step;
      progressEl.textContent = text;
    }
    if (!stepsEl) {
      return;
    }

    const sequence = ["camera", "detect", "liveness", "verify", "done"];
    const currentIdx = sequence.indexOf(step);
    stepsEl.querySelectorAll("li").forEach(function (node, idx) {
      node.classList.toggle("is-active", idx === currentIdx);
      node.classList.toggle("is-done", currentIdx > -1 && idx < currentIdx);
    });
  }

  function setError(message) {
    if (!errorEl) return;
    errorEl.hidden = !message;
    errorEl.textContent = message || "";
  }

  function clearOverlay() {
    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  function drawBoundingBox(bbox, state) {
    clearOverlay();
    if (!bbox || bbox.length !== 4) {
      return;
    }
    if (state === "detecting" || state === "no_face") {
      overlayContext.strokeStyle = "#ef4444";
    } else if (state === "processing" || state === "liveness") {
      overlayContext.strokeStyle = "#f59e0b";
    } else {
      overlayContext.strokeStyle = "#16a34a";
    }
    overlayContext.lineWidth = 3;
    overlayContext.strokeRect(bbox[0], bbox[1], bbox[2], bbox[3]);
  }

  async function initCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("Camera API is not available in this browser.");
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      STATE.stream = stream;
      videoEl.srcObject = stream;
      await videoEl.play();
      setError("");
      return true;
    } catch (_error) {
      setError("Camera permission denied. Please allow camera access and retry.");
      return false;
    }
  }

  function captureFrame() {
    if (!videoEl.videoWidth || !videoEl.videoHeight) {
      return null;
    }
    frameContext.drawImage(videoEl, 0, 0, frameCanvas.width, frameCanvas.height);
    return frameCanvas.toDataURL("image/jpeg", STATE.quality);
  }

  function mapStatusToUi(payload) {
    const status = payload.status || "unknown";
    statusEl.textContent = `Status: ${status}`;

    if (successEl) {
      successEl.hidden = true;
    }

    if (payload.prompt) {
      promptEl.textContent = payload.prompt;
      promptEl.hidden = false;
    } else {
      promptEl.textContent = "";
      promptEl.hidden = true;
    }

    if (status === "session_started") {
      setProgress("camera", "Camera ready");
      return;
    }
    if (status === "no_face") {
      setProgress("detect", "No face detected");
      setError("No face detected. Center your face inside the oval guide.");
      return;
    }
    if (status === "face_detected") {
      setProgress("detect", "Face detected");
      setError("");
      return;
    }
    if (status === "liveness_check") {
      setProgress("liveness", "Liveness check in progress");
      if (!payload.prompt) {
        promptEl.textContent = "Blink naturally";
        promptEl.hidden = false;
      }
      return;
    }
    if (status === "identity_verified") {
      setProgress("verify", "Identity verified");
      if (!payload.prompt) {
        promptEl.textContent = "Turn head left slowly";
        promptEl.hidden = false;
      }
      return;
    }
    if (status === "attendance_marked") {
      setProgress("done", "Attendance marked");
      if (successEl) {
        successEl.hidden = false;
      }
      setError("");
      promptEl.hidden = true;
      stopSendingFrames();
      return;
    }
    if (status === "invalid_frame") {
      setError("Could not process frame. Hold steady and try again.");
      return;
    }
    if (status === "verification_failed") {
      setProgress("verify", "Verification failed");
      setError(payload.message || "Verification failed. Please retry.");
      stopSendingFrames();
      return;
    }
    if (status === "timeout") {
      setProgress("timeout", "Verification timed out");
      setError("Session timed out. Click Start Verification to retry.");
      stopSendingFrames();
      return;
    }
    if (status === "cancelled") {
      setProgress("cancelled", "Verification cancelled");
      stopSendingFrames();
      return;
    }
    if (status === "unauthorized") {
      setError("You are not authorized. Please sign in again.");
      stopSendingFrames();
    }
  }

  function stopSendingFrames() {
    STATE.isVerifying = false;
    if (STATE.sendIntervalId) {
      clearInterval(STATE.sendIntervalId);
      STATE.sendIntervalId = null;
    }
  }

  function sendFrame() {
    if (!STATE.socket || !STATE.socket.connected || !STATE.isVerifying) {
      return;
    }
    const frame = captureFrame();
    if (frame) {
      STATE.socket.emit("frame", { frame: frame });
      drawBoundingBox([220, 110, 200, 260], "processing");
    }
  }

  function connectSocket() {
    STATE.socket = io({ reconnection: true, reconnectionAttempts: STATE.maxReconnectAttempts });

    STATE.socket.on("connect", function () {
      STATE.reconnectAttempt = 0;
      setError("");
    });

    STATE.socket.on("disconnect", function () {
      STATE.reconnectAttempt += 1;
      if (STATE.reconnectAttempt > STATE.maxReconnectAttempts) {
        setError("Connection lost. Refresh the page to reconnect.");
      }
    });

    STATE.socket.on("verification_status", function (payload) {
      if (payload.bbox) {
        let drawState = "ok";
        if (payload.status === "no_face") drawState = "no_face";
        else if (payload.status === "liveness_check") drawState = "liveness";
        else if (payload.status === "identity_verified") drawState = "processing";
        drawBoundingBox(payload.bbox, drawState);
      } else {
        clearOverlay();
      }
      mapStatusToUi(payload || {});
    });
  }

  function startVerification() {
    if (!STATE.socket || !STATE.socket.connected) {
      setError("Realtime connection unavailable. Please wait for reconnect.");
      return;
    }
    setError("");
    promptEl.textContent = "";
    STATE.isVerifying = true;
    setProgress("starting", "Starting verification");
    setError("");
    if (successEl) {
      successEl.hidden = true;
    }
    STATE.socket.emit("start_verification", {});

    stopSendingFrames();
    STATE.isVerifying = true;
    STATE.sendIntervalId = setInterval(sendFrame, Math.floor(1000 / STATE.fps));
  }

  function cancelVerification() {
    if (STATE.socket && STATE.socket.connected) {
      STATE.socket.emit("cancel_verification", {});
    }
    stopSendingFrames();
    clearOverlay();
  }

  function cleanupCamera() {
    if (STATE.stream) {
      STATE.stream.getTracks().forEach(function (track) {
        track.stop();
      });
      STATE.stream = null;
    }
  }

  startBtn.addEventListener("click", startVerification);
  cancelBtn.addEventListener("click", cancelVerification);

  window.addEventListener("beforeunload", function () {
    cancelVerification();
    cleanupCamera();
    if (STATE.socket) {
      STATE.socket.disconnect();
    }
  });

  initCamera();
  connectSocket();
})();