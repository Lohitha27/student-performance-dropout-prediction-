import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'attendance': [85, 60, 90, 50, 70, 95, 40, 88],
    'study_hours': [4, 2, 5, 1, 3, 6, 1, 4],
    'previous_marks': [78, 55, 88, 45, 65, 92, 40, 80],
    'family_support': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'dropout': ['No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical data
encoder = LabelEncoder()
df['family_support'] = encoder.fit_transform(df['family_support'])
df['dropout'] = encoder.fit_transform(df['dropout'])

# Split features and target
X = df.drop('dropout', axis=1)
y = df['dropout']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Results
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))
