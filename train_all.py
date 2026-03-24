import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

ASSESSMENTS = {
    "GAD-7":  {"train": "GAD-7_train.csv",  "test": "GAD-7_test.csv"},
    "PHQ-9":  {"train": "PHQ-9_train.csv",  "test": "PHQ-9_test.csv"},
    "PSS-10": {"train": "PSS-10_train.csv", "test": "PSS-10_test.csv"},
    "Y-BOCS": {"train": "Y-BOCS_train.csv", "test": "Y-BOCS_test.csv"},
    "MDQ":    {"train": "MDQ_train.csv",    "test": "MDQ_test.csv"},
}


for name, files in ASSESSMENTS.items():
    print(f"Training {name}...")

    train_df = pd.read_csv(files["train"])
    test_df  = pd.read_csv(files["test"])

    feat    = [c for c in train_df.columns if c.startswith("Q")]
    X_train = train_df[feat].astype(int)
    X_test  = test_df[feat].astype(int)

    le      = LabelEncoder()
    y_train = le.fit_transform(train_df["Label"].astype(str))
    y_test  = le.transform(test_df["Label"].astype(str))

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  Accuracy: {acc*100:.2f}%")

    prefix = name.replace("-", "-")
    joblib.dump(clf, f"{MODELS_DIR}/{name}_model.pkl")
    joblib.dump(le,  f"{MODELS_DIR}/{name}_encoder.pkl")
    json.dump(
        {int(i): c for i, c in enumerate(le.classes_)},
        open(f"{MODELS_DIR}/{name}_map.json", "w"),
        indent=2
    )
    print(f"  Saved to models/{name}_model.pkl\n")

print("All models trained successfully!")
