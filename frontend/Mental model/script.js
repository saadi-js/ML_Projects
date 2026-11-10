const API_BASE_URL = "http://127.0.0.1:8000";
const form = document.querySelector("#prediction-form");
const submitButton = document.querySelector("#submit-button");
const statusMessage = document.querySelector("#status");
const result = document.querySelector("#result");
const score = document.querySelector("#score");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  statusMessage.textContent = "";
  result.hidden = true;

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  for (const field of [
    "Age",
    "Daily_Unlocks",
  ]) {
    payload[field] = Number(payload[field]);
  }
  for (const field of [
    "Avg_Daily_Usage_Hours",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
  ]) {
    payload[field] = Number(payload[field]);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      const details = data.detail ? JSON.stringify(data.detail) : "The request was rejected.";
      throw new Error(details);
    }

    score.textContent = data.predected_mental_health_score;
    result.hidden = false;
  } catch (error) {
    statusMessage.textContent = `Unable to get a prediction: ${error.message}`;
  } finally {
    submitButton.disabled = false;
  }
});
