# Credit Card Fraud Detection

## Overview
This project uses machine learning to detect fraudulent credit card transactions.

Fraud detection is important for banks because it helps reduce financial losses and protect customers.

## Project Goal
Build a simple model that can identify suspicious transactions using historical transaction data.

## Tools & Technologies
- Python
- pandas
- scikit-learn

## Dataset
This project uses a public credit card transaction dataset commonly used for fraud detection research.

Target variable:
- 0 → normal transaction
- 1 → fraudulent transaction

## How It Works
1. Load transaction data
2. Split data into training and testing sets
3. Train a Logistic Regression model
4. Predict suspicious transactions
5. Evaluate model performance

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the program:

python fraud_detection.py

## Example Output
The model outputs:
- precision
- recall
- accuracy
- fraud probability for a sample transaction

## Why This Matters in Banking
Machine learning can help banks:

- detect fraud in real time
- reduce financial losses
- protect customers
- improve trust and security

## Future Improvements
- Handle imbalanced data more effectively
- Try advanced models (Random Forest, XGBoost)
- Deploy as a real-time monitoring API
- Integrate with streaming transaction systems# fraud-detection-project
