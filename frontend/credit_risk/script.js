const form = document.querySelector("#risk-form");
const resultPanel = document.querySelector("#result-panel");
const submitButton = form.querySelector(".submit-button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.querySelector(".button-label").textContent = "Assessing profile...";

  const formData = new FormData(form);
  const payload = {
    person_age: Number(formData.get("person_age")),
    person_income: Number(formData.get("person_income")),
    person_home_ownership: formData.get("person_home_ownership"),
    person_emp_length: Number(formData.get("person_emp_length")),
    loan_intent: formData.get("loan_intent"),
    loan_grade: formData.get("loan_grade"),
    loan_amnt: Number(formData.get("loan_amnt")),
    loan_int_rate: Number(formData.get("loan_int_rate")),
    loan_percent_income: Number(formData.get("loan_percent_income")),
    cb_person_default_on_file: formData.get("cb_person_default_on_file"),
    cb_person_cred_hist_length: Number(formData.get("cb_person_cred_hist_length"))
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Assessment failed.");
    renderResult(data);
  } catch (error) {
    resultPanel.innerHTML = `
      <div class="result-placeholder error-state">
        <div class="placeholder-icon">!</div>
        <p class="section-label">Unable to assess</p>
        <h2>Something went wrong</h2>
        <p>${escapeHtml(error.message)}</p>
      </div>`;
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector(".button-label").textContent = "Assess application";
  }
});

function renderResult(data) {
  const probability = Number(data.default_probabilty);
  const percentage = Math.round(probability * 100);
  const isHighRisk = Number(data.prediction) === 1;
  const resultLabel = data.Result || (isHighRisk ? "High Risk" : "Low Risk");

  resultPanel.className = "card result-card result-ready";
  resultPanel.innerHTML = `
    <div>
      <p class="section-label">${isHighRisk ? "Review recommended" : "Assessment complete"}</p>
      <div class="risk-ring" style="--score: ${percentage}%">
        <div><div class="risk-score">${percentage}<span>%</span></div><small>RISK SCORE</small></div>
      </div>
      <div class="risk-result ${isHighRisk ? "high" : "low"}">${escapeHtml(resultLabel)}</div>
      <p class="risk-detail">${isHighRisk ? "The application is above the model's risk threshold." : "The application is below the model's risk threshold."}</p>
      <p class="threshold-note">Model threshold: ${Number(data.threshold).toFixed(2)}</p>
    </div>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[character]));
}