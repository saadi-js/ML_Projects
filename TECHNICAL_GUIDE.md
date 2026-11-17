# MentalModel Technical Guide

This guide explains the main technical concepts used across the three projects.
Examples are intentionally small, but they follow the patterns used in the
notebooks and FastAPI services.

## 1. Supervised learning

**Definition:** Supervised learning learns a relationship between input features
`X` and a known target `y`.

- Regression predicts a number, such as `Mental_Health_Score`.
- Classification predicts a class or class probability, such as an emotion or
  default risk.

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 2. Train-test split

**Definition:** A train-test split keeps evaluation data separate from the data
used to fit the model. This estimates how the model behaves on unseen examples.

The mental-health notebook uses a 70/30 split. Classification projects use
stratification where preserving class proportions matters.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
```

## 3. Data preprocessing

**Definition:** Preprocessing converts raw values into a form a model can use.
Typical operations include missing-value imputation, scaling, and categorical
encoding.

### Imputation

Imputation replaces missing values. Median imputation is robust to outliers.

```python
from sklearn.impute import SimpleImputer

numeric_imputer = SimpleImputer(strategy="median")
```

### Standardization

Standardization centers a numeric feature around zero and scales it to unit
variance. It is useful for models whose coefficients depend on feature scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

### One-hot encoding

One-hot encoding turns categories into binary columns. `handle_unknown="ignore"`
prevents an unseen category at inference time from crashing the request.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

### Ordinal encoding

Ordinal encoding is appropriate when categories have a meaningful order. The
mental-health project maps stress from low to very high.

```python
from sklearn.preprocessing import OrdinalEncoder

stress = OrdinalEncoder(
    categories=[["Low", "Medium", "High", "Very High"]]
)
```

### Log transformation

`log1p(x)` reduces the effect of strong right skew while remaining defined at
zero. The mental-health notebook applies it to `Study_Hours`.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transform = FunctionTransformer(np.log1p)
```

## 4. Pipeline and ColumnTransformer

**Definition:** A pipeline chains transformations and a final estimator. A
`ColumnTransformer` applies different pipelines to different columns.

This prevents training and serving from using different preprocessing rules.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

Every `Pipeline` step must be a two-item tuple:
`("step_name", estimator)`.

## 5. Regression

**Definition:** Regression predicts a continuous numeric value.

The mental-health project uses a `LinearRegression` baseline and searches a
`RandomForestRegressor` pipeline for a stronger model.

```python
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
)
```

### Regression metrics

- **MAE:** average absolute error; easier to interpret in target units.
- **MSE:** average squared error; penalizes larger errors more strongly.
- **R²:** proportion of target variance explained by the model.

```python
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
```

## 6. Classification

**Definition:** Classification predicts discrete labels. The emotion model has
six classes; the credit model has two.

`predict_proba` returns probabilities, which are useful when a product needs
confidence or a business threshold rather than only a class label.

```python
probabilities = classifier.predict_proba(X_test)
class_ids = probabilities.argmax(axis=1)
```

## 7. Class imbalance

**Definition:** Class imbalance occurs when one target class is much more common
than another. Accuracy can then hide poor minority-class performance.

The credit notebook computes a negative-to-positive ratio and passes it to
XGBoost as `scale_pos_weight`. Logistic regression also uses
`class_weight="balanced"`.

```python
negative, positive = np.bincount(y_train)
scale_weight = negative / positive
```

Useful metrics include precision, recall, F1, ROC-AUC, and the precision-recall
curve.

- **Precision:** of predicted positives, how many were truly positive?
- **Recall:** of actual positives, how many were found?
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking quality across classification thresholds.

## 8. Cross-validation and hyperparameter search

**Definition:** Cross-validation repeatedly trains and evaluates a model on
different folds. It gives a more stable estimate than one split.

`StratifiedKFold` preserves class proportions in classification folds. The
mental-health and credit notebooks also use `RandomizedSearchCV` to sample
hyperparameter combinations.

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=model,
    param_distributions={"classifier__max_depth": [3, 5, 7]},
    n_iter=3,
    cv=cv,
    scoring="roc_auc",
    random_state=42,
)
```

The `classifier__` or `regressor__` prefix addresses a parameter inside a
pipeline step.

## 9. Probability calibration and decision thresholds

**Definition:** Calibration improves how closely predicted probabilities
correspond to observed frequencies. A threshold converts a probability into a
business decision.

The credit project saves a calibrated XGBoost model and a separately selected
threshold. Therefore, a `Low Risk` result means the predicted probability is
below that stored threshold; it is not a guaranteed loan approval.

```python
probability = model.predict_proba(input_data)[:, 1][0]
prediction = int(probability > threshold)
```

Changing the threshold trades off false positives and false negatives. The
credit notebook uses precision-recall analysis to examine this trade-off.

## 10. Tokenization and padding

**Definition:** Neural text models consume numeric sequences, not raw strings.
Tokenization maps words to integer IDs. Padding makes sequences the same length.

The emotion project uses a 10,000-word vocabulary, an out-of-vocabulary token,
and a fixed sequence length of 50.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(["I feel happy"])
padded = pad_sequences(sequences, maxlen=50, padding="post", truncating="post")
```

The same fitted tokenizer must be used during inference. A different tokenizer
would produce different token IDs and invalidate the trained model.

## 11. Embeddings and recurrent neural networks

**Embedding:** An embedding layer learns a dense vector representation for each
token.

**RNN:** Processes a sequence recurrently and can model order-dependent patterns.

**LSTM:** Adds gates and memory designed to preserve useful long-range context.

**GRU:** Uses a simpler gated design than LSTM. The deployed emotion model uses
two bidirectional GRU layers, dropout, and a softmax output layer.

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Embedding, Dropout, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=300, input_length=50),
    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(GRU(64)),
    Dense(6, activation="softmax"),
])
```

**Softmax:** Converts the six output logits into probabilities that sum to one.
The largest probability determines the predicted emotion, and its value is
reported as confidence.

## 12. Text normalization

**Definition:** Normalization makes text consistent before tokenization. The
emotion API lowercases text, removes apostrophes and non-alphanumeric symbols,
and collapses repeated whitespace.

```python
text = text.lower()
text = re.sub("'", "", text)
text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
text = re.sub(r"\s+", " ", text).strip()
```

Training and serving should use compatible normalization rules.

## 13. Persisting models

**Definition:** Model persistence stores a fitted model so the API does not
retrain it for every request.

The repository uses Joblib for scikit-learn/XGBoost artifacts and Keras's
`.keras` format for the neural model.

```python
import joblib

joblib.dump(fitted_pipeline, "credit_model.pkl")
loaded_pipeline = joblib.load("credit_model.pkl")
```

The emotion service separately persists the Keras model and tokenizer because
both are required for inference.

## 14. FastAPI and Pydantic

**FastAPI:** A Python web framework used to expose prediction endpoints and
interactive OpenAPI documentation.

**Pydantic model:** A typed request schema that validates and documents JSON
input.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EmotionRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: EmotionRequest):
    return {"text": request.text}
```

FastAPI provides Swagger UI at `/docs` and the OpenAPI schema at
`/openapi.json`.

## 15. Static files and browser integration

**Definition:** Static files are frontend assets served directly without
server-side rendering. The credit-risk backend mounts its HTML, CSS, and
JavaScript folder at `/`.

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles

frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "credit_risk"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
```

The browser sends the form as JSON to the API:

```javascript
const response = await fetch("/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});
const result = await response.json();
```

Mount the static frontend after registering API routes so `/predict` remains
available.

## 16. JSON serialization at the API boundary

NumPy scalar types are not always JSON serializable by FastAPI. Convert model
outputs to native Python values before returning them.

```python
return {
    "probability": float(probability),
    "prediction": int(prediction),
    "threshold": float(threshold),
}
```

This is especially important for values returned by NumPy arrays,
scikit-learn, and XGBoost.

## 17. End-to-end inference pattern

All three applications follow the same serving pattern:

1. Validate JSON input.
2. Reproduce the training-time preprocessing.
3. Run the persisted model.
4. Convert outputs into JSON-safe values.
5. Return a small response used by the browser UI.

```python
@app.post("/predict")
def predict(request: RequestSchema):
    frame = pd.DataFrame([request.model_dump()])
    probability = model.predict_proba(frame)[:, 1][0]
    label = int(probability > threshold)
    return {"probability": float(probability), "label": label}
```

## 18. Responsible interpretation

These applications provide model predictions, not professional conclusions:

- A mental-health score is not a clinical diagnosis.
- An emotion label is not a psychological assessment.
- A credit-risk label is not an approval or rejection decision.

Production use should add authentication, restricted CORS, HTTPS, monitoring,
input auditing, model/version tracking, fairness evaluation, and domain-expert
review.
