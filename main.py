from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="Mental Health Assessment API",
    description="API for mental health assessment using GAD-7, PHQ-9, PSS-10, Y-BOCS, MDQ",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSESSMENTS = {
    "GAD-7": {
        "train_csv": "GAD7_train.csv",
        "test_csv":  "GAD7_test.csv",
        "num_questions": 7,
        "max_answer": 3,
        "model_file":   "GAD-7_model.pkl",
        "encoder_file": "GAD-7_encoder.pkl",
        "map_file":     "GAD-7_map.json",
        "reverse_indices": [],
        "interpret": lambda s: (
            {"level": "minimal",  "label": "Minimal Anxiety",  "color": "green"}  if s <= 4  else
            {"level": "mild",     "label": "Mild Anxiety",     "color": "yellow"} if s <= 9  else
            {"level": "moderate", "label": "Moderate Anxiety", "color": "orange"} if s <= 14 else
            {"level": "severe",   "label": "Severe Anxiety",   "color": "red"}
        ),
    },
    "PHQ-9": {
        "train_csv": "PHQ9_train.csv",
        "test_csv":  "PHQ9_test.csv",
        "num_questions": 9,
        "max_answer": 3,
        "model_file":   "PHQ-9_model.pkl",
        "encoder_file": "PHQ-9_encoder.pkl",
        "map_file":     "PHQ-9_map.json",
        "reverse_indices": [],
        "interpret": lambda s: (
            {"level": "minimal",           "label": "Minimal Depression",           "color": "green"}  if s <= 4  else
            {"level": "mild",              "label": "Mild Depression",              "color": "yellow"} if s <= 9  else
            {"level": "moderate",          "label": "Moderate Depression",          "color": "orange"} if s <= 14 else
            {"level": "moderately_severe", "label": "Moderately Severe Depression", "color": "red"}    if s <= 19 else
            {"level": "severe",            "label": "Severe Depression",            "color": "red"}
        ),
    },
    "PSS-10": {
        "train_csv": "PSS10_train.csv",
        "test_csv":  "PSS10_test.csv",
        "num_questions": 10,
        "max_answer": 4,
        "model_file":   "PSS-10_model.pkl",
        "encoder_file": "PSS-10_encoder.pkl",
        "map_file":     "PSS-10_map.json",
        "reverse_indices": [3, 4, 5, 6, 8],
        "interpret": lambda s: (
            {"level": "low",    "label": "Low Stress",    "color": "green"}  if s <= 13 else
            {"level": "medium", "label": "Medium Stress", "color": "orange"} if s <= 26 else
            {"level": "high",   "label": "High Stress",   "color": "red"}
        ),
    },
    "Y-BOCS": {
        "train_csv": "YBOCS_train.csv",
        "test_csv":  "YBOCS_test.csv",
        "num_questions": 10,
        "max_answer": 4,
        "model_file":   "Y-BOCS_model.pkl",
        "encoder_file": "Y-BOCS_encoder.pkl",
        "map_file":     "Y-BOCS_map.json",
        "reverse_indices": [],
        "interpret": lambda s: (
            {"level": "subclinical", "label": "Subclinical OCD", "color": "green"}  if s <= 7  else
            {"level": "mild",        "label": "Mild OCD",        "color": "yellow"} if s <= 15 else
            {"level": "moderate",    "label": "Moderate OCD",    "color": "orange"} if s <= 23 else
            {"level": "severe",      "label": "Severe OCD",      "color": "red"}    if s <= 31 else
            {"level": "extreme",     "label": "Extreme OCD",     "color": "red"}
        ),
    },
    "MDQ": {
        "train_csv": "MDQ_train.csv",
        "test_csv":  "MDQ_test.csv",
        "num_questions": 13,
        "max_answer": 1,
        "model_file":   "MDQ_model.pkl",
        "encoder_file": "MDQ_encoder.pkl",
        "map_file":     "MDQ_map.json",
        "reverse_indices": [],
        "interpret": lambda s: (
            {"level": "unlikely", "label": "Bipolar Disorder Unlikely", "color": "green"} if s < 7 else
            {"level": "likely",   "label": "Bipolar Disorder Likely",   "color": "red"}
        ),
    },
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_model(assessment_key: str, data_dir: str = ".") -> Dict[str, Any]:
    cfg = ASSESSMENTS[assessment_key]

    train_df = pd.read_csv(os.path.join(data_dir, cfg["train_csv"]))
    test_df  = pd.read_csv(os.path.join(data_dir, cfg["test_csv"]))

    feat = [c for c in train_df.columns if c.startswith("Q")]
    X_train, y_train = train_df[feat].astype(int), train_df["Label"].astype(str)
    X_test,  y_test  = test_df[feat].astype(int),  test_df["Label"].astype(str)

    le   = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_tr)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_te, clf.predict(X_test))

    joblib.dump(clf, os.path.join(MODELS_DIR, cfg["model_file"]))
    joblib.dump(le,  os.path.join(MODELS_DIR, cfg["encoder_file"]))
    label_map = {int(i): c for i, c in enumerate(le.classes_)}
    with open(os.path.join(MODELS_DIR, cfg["map_file"]), "w") as f:
        json.dump(label_map, f, indent=2)

    return {"assessment": assessment_key, "accuracy": round(acc * 100, 2), "classes": list(le.classes_)}


def load_model(assessment_key: str):
    cfg = ASSESSMENTS[assessment_key]
    mf  = os.path.join(MODELS_DIR, cfg["model_file"])
    ef  = os.path.join(MODELS_DIR, cfg["encoder_file"])
    jf  = os.path.join(MODELS_DIR, cfg["map_file"])
    if not all(os.path.exists(p) for p in [mf, ef, jf]):
        return None, None, None
    clf = joblib.load(mf)
    le  = joblib.load(ef)
    with open(jf) as f:
        label_map = json.load(f)
    return clf, le, label_map


class PredictRequest(BaseModel):
    answers: List[int] = Field(..., description="List of answers in order (starting from 0)")

    class Config:
        schema_extra = {"example": {"answers": [1, 2, 0, 1, 3, 2, 1]}}


class TrainRequest(BaseModel):
    data_dir: str = Field(".", description="Path containing the CSV files")
    assessments: Optional[List[str]] = Field(
        None, description="List of assessment keys to train (leave empty for all)"
    )


def _predict(assessment_key: str, req: PredictRequest):
    cfg = ASSESSMENTS[assessment_key]
    n   = cfg["num_questions"]

    if len(req.answers) != n:
        raise HTTPException(
            status_code=422,
            detail=f"{assessment_key} requires {n} answers, got {len(req.answers)}"
        )

    for i, a in enumerate(req.answers):
        if not (0 <= a <= cfg["max_answer"]):
            raise HTTPException(
                status_code=422,
                detail=f"Answer {i+1} is out of range (0–{cfg['max_answer']})"
            )

    answers = list(req.answers)
    for i in cfg["reverse_indices"]:
        if i < len(answers):
            answers[i] = cfg["max_answer"] - answers[i]

    total_score    = sum(answers)
    max_possible   = n * cfg["max_answer"]
    interpretation = cfg["interpret"](total_score)

    clf, le, label_map = load_model(assessment_key)
    model_result = None
    if clf is not None:
        df         = pd.DataFrame([answers], columns=[f"Q{i+1}" for i in range(n)])
        pred_enc   = clf.predict(df)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        model_result = {
            "predicted_class": str(pred_label),
            "label_meaning": label_map.get(int(pred_enc), "unknown"),
        }

    return {
        "assessment": assessment_key,
        "total_score": total_score,
        "max_possible": max_possible,
        "percentage": round(total_score / max_possible * 100, 1),
        "interpretation": interpretation,
        "model_prediction": model_result,
        "note": "This tool is for educational and research purposes only.",
    }


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Mental Health Assessment API",
        "docs": "/docs",
        "available_assessments": list(ASSESSMENTS.keys()),
    }


@app.get("/health", tags=["General"])
def health():
    status = {}
    for key, cfg in ASSESSMENTS.items():
        mf = os.path.join(MODELS_DIR, cfg["model_file"])
        status[key] = "ready" if os.path.exists(mf) else "not_trained"
    return {"status": "ok", "models": status}


@app.post("/predict/gad7", tags=["Predict"], summary="GAD-7 – Generalized Anxiety Disorder")
def predict_gad7(req: PredictRequest):
    return _predict("GAD-7", req)


@app.post("/predict/phq9", tags=["Predict"], summary="PHQ-9 – Depression")
def predict_phq9(req: PredictRequest):
    return _predict("PHQ-9", req)


@app.post("/predict/pss10", tags=["Predict"], summary="PSS-10 – Perceived Stress Scale")
def predict_pss10(req: PredictRequest):
    return _predict("PSS-10", req)


@app.post("/predict/ybocs", tags=["Predict"], summary="Y-BOCS – OCD")
def predict_ybocs(req: PredictRequest):
    return _predict("Y-BOCS", req)


@app.post("/predict/mdq", tags=["Predict"], summary="MDQ – Bipolar Disorder")
def predict_mdq(req: PredictRequest):
    return _predict("MDQ", req)


@app.post("/train", tags=["Train"], summary="Train models from CSV files")
def train(req: TrainRequest):
    keys    = req.assessments or list(ASSESSMENTS.keys())
    results = []
    errors  = []

    for key in keys:
        if key not in ASSESSMENTS:
            errors.append({"assessment": key, "error": "Unknown assessment key"})
            continue
        try:
            r = train_model(key, req.data_dir)
            results.append(r)
        except Exception as e:
            errors.append({"assessment": key, "error": str(e)})

    return {"trained": results, "errors": errors}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
