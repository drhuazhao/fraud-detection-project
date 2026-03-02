import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def main():

  
    df = pd.read_csv("creditcard_sample.csv")

   
    df = df.sample(min(5000, len(df)), random_state=42)


   
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
 
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)


    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of fraud

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))


    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"=== ROC-AUC ===\n{auc:.4f}")
    except ValueError:
        # This can happen if the test set has only one class after sampling
        print("=== ROC-AUC ===\nNot available (only one class in y_test).")


    sample_score = y_proba[0]
    print(f"\nExample transaction fraud probability: {sample_score:.4f}")


if __name__ == "__main__":
    main()
