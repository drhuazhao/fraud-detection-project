import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def main():
    # 1) Load data
    # Make sure creditcard.csv is in the same folder as this file
    df = pd.read_csv("creditcard_sample.csv")

    # Optional: use a smaller sample so it runs fast on any laptop
    df = df.sample(5000, random_state=42)

    # 2) Split features/target
    # In this dataset: Class = 1 (fraud), 0 (normal)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 3) Train/test split (stratify keeps fraud ratio similar in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4) Scale features (Logistic Regression works better with scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Train model
    # class_weight='balanced' helps because fraud is rare (imbalanced dataset)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # 6) Predict & evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of fraud

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # ROC-AUC is useful for imbalanced problems
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"=== ROC-AUC ===\n{auc:.4f}")
    except ValueError:
        # This can happen if the test set has only one class after sampling
        print("=== ROC-AUC ===\nNot available (only one class in y_test).")

    # 7) Demo: score one transaction
    sample_score = y_proba[0]
    print(f"\nExample transaction fraud probability: {sample_score:.4f}")


if __name__ == "__main__":
    main()