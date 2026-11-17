# MentalModel

MentalModel is a collection of three machine-learning applications exposed through
FastAPI services and lightweight browser interfaces:

1. **Mental health score prediction** - estimates a continuous student wellbeing score.
2. **Text emotion detection** - classifies text into one of six emotions.
3. **Credit risk assessment** - estimates loan-default risk and compares it with a
   calibrated decision threshold.

The project is organized as a small model-serving portfolio. Each project has a
training notebook, a persisted model artifact, a FastAPI backend, and (where
implemented) a browser frontend.

## Repository layout

```text
MentalModel/
├── backend/
│   ├── Mental_model.py       # Mental-health prediction API
│   ├── emotions.py           # Text-emotion prediction API
│   └── credit_risk.py        # Credit-risk prediction API and frontend mount
├── frontend/
│   ├── Mental model/         # Mental-health form
│   ├── Text_model/           # Emotion-analysis interface
│   └── credit_risk/          # Credit-risk dashboard
├── model/
│   ├── Mental_health_Model.pkl
│   ├── emotion_model/
│   │   ├── BiGRU_Model.keras
│   │   └── tokenizer.pkl
│   └── credit_risk/
│       ├── credit_model.pkl
│       └── threshold.pkl
└── notebooks/
    ├── Mental_HScore.ipynb
    ├── Emotion_analysis.ipynb
    └── creditRisk.ipynb
```

The repository currently does not include a dependency manifest. The imports used
by the applications require Python, FastAPI, Uvicorn, Pydantic, pandas, NumPy,
scikit-learn, Joblib, XGBoost, TensorFlow/Keras, and Hugging Face Datasets.

## Project 1: Mental health score prediction

### Purpose

The application predicts `Mental_Health_Score` from student demographic,
academic, social-media, and wellbeing features. This is a **regression** problem
because the target is numeric and continuous.

### Data and methodology

The notebook reads the student social-media and mental-health dataset, groups
countries into common categories plus `Other`, and separates the target from the
features. The feature preparation includes:

- `Study_Hours`: `log1p` transformation followed by standardization.
- Other numeric variables: standardization.
- `Stress_Level`: ordinal encoding in the order `Low`, `Medium`, `High`,
  `Very High`.
- Categorical variables: one-hot encoding with unknown values ignored.
- `ColumnTransformer` and `Pipeline`: preprocessing and regression remain together
  during training and inference.
- A 70/30 train-test split with `random_state=42`.
- A baseline `LinearRegression` model.
- `RandomizedSearchCV` over a `RandomForestRegressor` pipeline using five-fold
  cross-validation and R² scoring.
- The selected search configuration uses 200 trees, depth 15, minimum leaf size
  2, and minimum split size 5.

The selected random-forest pipeline is saved as `model/Mental_health_Model.pkl`.
The notebook records a baseline test R² of approximately `0.740` before the
random-forest search.

### API and UI

`backend/Mental_model.py` validates the request with Pydantic, prepares the
`Grouped_Country` feature, calls the persisted model, and returns a rounded
`predected_mental_health_score` value. The form is in `frontend/Mental model/`.

Run it from the repository root:

```powershell
uvicorn backend.Mental_model:app --reload --port 8001
```

Then open `http://127.0.0.1:8001/docs`.

## Project 2: Text emotion detection

### Purpose

The application classifies text into:

```text
sadness, joy, love, anger, fear, surprise
```

This is a **multi-class text-classification** problem.

### Data and methodology

The notebook loads the `dair-ai/emotion` dataset using Hugging Face Datasets and
keeps its train/test splits. The text workflow is:

1. Fit a Keras `Tokenizer` with a 10,000-word vocabulary and an `<OOV>` token.
2. Convert text to integer sequences.
3. Pad or truncate every sequence to 50 tokens.
4. Train and compare RNN, LSTM, GRU, and bidirectional-GRU models.
5. Use sparse categorical cross-entropy and Adam optimization.
6. Evaluate with test loss, accuracy, a confusion matrix, and a classification
   report.

The notebook also computes class weights and uses early stopping based on
validation loss while comparing the recurrent architectures. The deployed model
is a bidirectional GRU with 300-dimensional embeddings, GRU layers of 128 and
64 units, and dropout.

The persisted production artifact is the bidirectional GRU model
`model/emotion_model/BiGRU_Model.keras`, together with its tokenizer
`model/emotion_model/tokenizer.pkl`.

### API and UI

`backend/emotions.py` cleans incoming text, tokenizes it, pads it to 50 tokens,
and returns the predicted emotion, confidence, and all class probabilities. It
also exposes `/health` and serves the frontend in `frontend/Text_model/`.

Run it with:

```powershell
uvicorn backend.emotions:app --reload --port 8002
```

Open `http://127.0.0.1:8002/`.

## Project 3: Credit risk assessment

### Purpose

The application estimates the probability of loan default and labels an
application as `High Risk` or `Low Risk`. This is a **binary classification**
problem. `Low Risk` means the predicted probability is below the model's stored
threshold; it is decision support, not an automatic loan approval.

### Data and methodology

The notebook reads `credit_risk_dataset.csv`, separates `loan_status` as the
target, and uses:

- Median imputation and standardization for numeric features in logistic
  regression.
- Median imputation for numeric features in XGBoost.
- Constant-value imputation plus one-hot encoding for categorical features.
- `StratifiedKFold` with five folds.
- Accuracy, precision, recall, F1, and ROC-AUC scoring.
- A balanced logistic-regression pipeline.
- An XGBoost classifier with class-imbalance weighting through
  `scale_pos_weight`.
- Probability calibration and precision-recall analysis.
- A saved threshold chosen after calibration.

The notebook reports the stronger XGBoost validation results as approximately
`0.9471` ROC-AUC, `0.9180` accuracy, and `0.8073` F1.

The deployed artifacts are `model/credit_risk/credit_model.pkl` and
`model/credit_risk/threshold.pkl`.

### API and UI

`backend/credit_risk.py` validates the 11 loan features, loads the model and
threshold, returns the default probability and risk label, and mounts
`frontend/credit_risk/` as the root static site. The frontend submits JSON to
`POST /predict`.

Run it with:

```powershell
uvicorn backend.credit_risk:app --reload
```

Open `http://127.0.0.1:8000/`.

## Common API examples

### Mental health

```json
{
  "Age": 21,
  "Gender": "Female",
  "Country": "India",
  "Academic_Level": "Undergraduate",
  "Most_Used_Platform": "Instagram",
  "Purpose_Of_Use": "Education",
  "Avg_Daily_Usage_Hours": 4.5,
  "Daily_Unlocks": 120,
  "Study_Hours": 5,
  "Physical_Activity_Hours": 1.5,
  "Sleep_Hours_Per_Night": 7,
  "Stress_Level": "Medium"
}
```

### Emotion

```json
{
  "text": "Today feels wonderful."
}
```

### Credit risk

```json
{
  "person_age": 28,
  "person_income": 65000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 12.5,
  "loan_percent_income": 0.15,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

## Important operational notes

- Model files must remain available at the paths used by each backend.
- `Mental_model.py` currently uses a machine-specific absolute model path; use a
  project-relative `Path` before deploying to another machine.
- The mental-health API computes a grouped-country value but currently places the
  raw country in `Grouped_Country`; align this with the training transformation
  before production deployment.
- The APIs load model artifacts at import/startup time, so a missing or
  incompatible artifact prevents the service from starting.
- The model outputs are predictions, not medical diagnoses, emotional diagnoses,
  or guaranteed lending decisions.
- CORS is permissive in the current services and should be restricted for
  production deployment.

For definitions, implementation patterns, and small code examples, see
[TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md).
