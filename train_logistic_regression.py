import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_PATH = "data/cleaned_dataset.csv"

LOGISTIC_MODEL_PATH = "models/ddos_logistic_regression.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

LABEL_COL = "Label"


def print_results(model_name, y_test, predictions, label_encoder):
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    ))


df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nLabel counts:")
print(df[LABEL_COL].value_counts())


# Split into input features and the thing we want to predict
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]


# Convert BENIGN / DDoS labels into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Keep the same label ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Baseline model
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

print_results(
    "Logistic Regression Results",
    y_test,
    logistic_predictions,
    label_encoder
)


# Save the baseline model and label encoder
os.makedirs("models", exist_ok=True)

joblib.dump(logistic_model, LOGISTIC_MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print("\nSaved logistic regression model to:", LOGISTIC_MODEL_PATH)
print("Saved label encoder to:", ENCODER_PATH)