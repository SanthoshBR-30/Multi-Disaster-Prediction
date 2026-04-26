const API = window.location.origin + "/predict";
    let selectedFile = null;

const toggleBtn = document.getElementById("toggle");

toggleBtn.addEventListener("click", () => {
    document.body.classList.toggle("dark");
});

    // ── Drop zone logic ───────────────────────────────────────────────────────
    const dropZone  = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", e => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

    function handleFile(file) {
      if (!file.type.startsWith("image/")) { showError("Please select an image file."); return; }
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = e => {
        document.getElementById("preview-img").src = e.target.result;
        document.getElementById("preview-wrap").style.display = "block";
        dropZone.style.display = "none";
        document.getElementById("classify-btn").disabled = false;
        document.getElementById("btn-text").textContent = "Classify Image";
        hideError();
        document.getElementById("result-box").style.display = "none";
      };
      reader.readAsDataURL(file);
    }

    function resetUI() {
      selectedFile = null;
      fileInput.value = "";
      document.getElementById("preview-wrap").style.display = "none";
      dropZone.style.display = "block";
      document.getElementById("classify-btn").disabled = true;
      document.getElementById("btn-text").textContent = "Select an image first";
      document.getElementById("result-box").style.display = "none";
      hideError();
    }

    // ── Classify ──────────────────────────────────────────────────────────────
    async function classify() {
      if (!selectedFile) return;

      setLoading(true);
      hideError();
      document.getElementById("result-box").style.display = "none";

      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const res  = await fetch(API, { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Server error");
        showResults(data);
      } catch (err) {
        showError(err.message.includes("Failed to fetch")
          ? "Cannot reach backend. Make sure app.py is running on port 5000."
          : err.message);
      } finally {
        setLoading(false);
      }
    }

    function showResults(data) {
      document.getElementById("pred-class").textContent = data.predicted_class;
      document.getElementById("pred-conf").textContent =
        `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

      // Sort probabilities descending
      const sorted = Object.entries(data.all_probabilities)
        .sort((a, b) => b[1] - a[1]);

      const barsDiv = document.getElementById("bars");
      barsDiv.innerHTML = "";
      sorted.forEach(([cls, prob]) => {
        const pct = (prob * 100).toFixed(1);
        barsDiv.innerHTML += `
          <div class="bar-row">
            <div class="bar-label"><span>${cls}</span><span>${pct}%</span></div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct}%"></div>
            </div>
          </div>`;
      });

      document.getElementById("result-box").style.display = "block";
    }

    function setLoading(on) {
      const btn     = document.getElementById("classify-btn");
      const txt     = document.getElementById("btn-text");
      const spinner = document.getElementById("spinner");
      btn.disabled        = on;
      txt.style.display   = on ? "none" : "inline";
      spinner.style.display = on ? "block" : "none";
    }

    function showError(msg) {
      const box = document.getElementById("error-box");
      box.textContent = "⚠ " + msg;
      box.style.display = "block";
    }
    function hideError() {
      document.getElementById("error-box").style.display = "none";
    }