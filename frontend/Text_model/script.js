const message = document.querySelector("#message");
const charCount = document.querySelector("#char-count");
const analyzeButton = document.querySelector("#analyze-button");
const errorMessage = document.querySelector("#error-message");
const emptyState = document.querySelector("#empty-state");
const loadingState = document.querySelector("#loading-state");
const resultContent = document.querySelector("#result-content");
const historySection = document.querySelector("#history-section");
const history = document.querySelector("#history");
const emotionColors = {
    sadness: "#72a8ff", joy: "#ffc46b", love: "#f28fc8",
    anger: "#ff7d8c", fear: "#b492ff", surprise: "#5edbd1"
};
const emotionEmoji = {
    sadness: "☾", joy: "☀", love: "♥", anger: "⚡", fear: "◌", surprise: "✦"
};

message.addEventListener("input", () => {
    charCount.textContent = message.value.length;
});

document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        message.value = chip.textContent;
        message.dispatchEvent(new Event("input"));
        message.focus();
    });
});

document.querySelector("#theme-toggle").addEventListener("click", (event) => {
    document.body.classList.toggle("light");
    event.currentTarget.textContent = document.body.classList.contains("light") ? "☾" : "☼";
});

analyzeButton.addEventListener("click", analyze);
message.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
});

async function analyze() {
    const text = message.value.trim();
    if (!text) {
        showError("Write a message first so I can read its signal.");
        message.focus();
        return;
    }

    errorMessage.textContent = "";
    analyzeButton.disabled = true;
    emptyState.classList.add("hidden");
    resultContent.classList.add("hidden");
    loadingState.classList.remove("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        if (!response.ok) {
            let detail = "The model could not analyze that message.";
            try { detail = (await response.json()).detail || detail; } catch (_) { /* use default */ }
            throw new Error(detail);
        }
        renderResult(await response.json());
        saveHistory({ text, emotion: resultContent.dataset.emotion, confidence: resultContent.dataset.confidence });
    } catch (error) {
        loadingState.classList.add("hidden");
        emptyState.classList.remove("hidden");
        showError(error.message || "Unable to connect to the emotion model.");
    } finally {
        analyzeButton.disabled = false;
    }
}

function renderResult(data) {
    const confidence = Number(data.confidence);
    const emotion = data.emotion.toLowerCase();
    resultContent.dataset.emotion = emotion;
    resultContent.dataset.confidence = confidence;
    document.querySelector("#emotion-name").textContent = emotion;
    document.querySelector("#emotion-emoji").textContent = emotionEmoji[emotion] || "✦";
    document.querySelector("#confidence").textContent = `${(confidence * 100).toFixed(1)}%`;
    document.querySelector("#result-time").textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    document.querySelector("#emotion-orb").style.background = `radial-gradient(circle at 35% 30%, #fff, ${emotionColors[emotion] || "#9a7cff"} 35%, #271e68)`;
    document.querySelector("#confidence-fill").style.width = `${confidence * 100}%`;
    const probabilities = document.querySelector("#probabilities");
    probabilities.innerHTML = "";
    Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]).forEach(([name, value]) => {
        const row = document.createElement("div");
        row.className = "probability";
        row.innerHTML = `<span>${name}</span><div class="bar"><i style="--bar-color:${emotionColors[name] || "#9a7cff"}; width:${value * 100}%"></i></div><span>${(value * 100).toFixed(1)}%</span>`;
        probabilities.appendChild(row);
    });
    loadingState.classList.add("hidden");
    resultContent.classList.remove("hidden");
}

function showError(messageText) {
    errorMessage.textContent = messageText;
}

function saveHistory(item) {
    const items = JSON.parse(localStorage.getItem("pulse-history") || "[]");
    items.unshift(item);
    localStorage.setItem("pulse-history", JSON.stringify(items.slice(0, 6)));
    renderHistory();
}

function renderHistory() {
    const items = JSON.parse(localStorage.getItem("pulse-history") || "[]");
    historySection.classList.toggle("hidden", items.length === 0);
    history.innerHTML = items.map((item) => `
        <article class="history-card">
            <header><strong>${escapeHtml(item.emotion)}</strong><span>${(Number(item.confidence) * 100).toFixed(0)}%</span></header>
            <p title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</p>
        </article>`).join("");
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
    }[character]));
}

document.querySelector("#clear-history").addEventListener("click", () => {
    localStorage.removeItem("pulse-history");
    renderHistory();
});
renderHistory();