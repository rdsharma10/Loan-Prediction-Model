"""
Scikit-Learn (sklearn) Cheatsheet
Author: Rishi Sharma
Description: This file contains a comprehensive Scikit-Learn (sklearn) cheatsheet covering essential functions and operations.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("sample_data.csv")  # Replace with your dataset
X = data.drop("target", axis=1)
y = data["target"]

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encoding Categorical Data
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Handling Missing Data
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_reg = lin_reg.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_reg))
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_reg))

# Classification Model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train_encoded)
y_pred_log = log_reg.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test_encoded, y_pred_log))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred_log))

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Support Vector Machine
svm = SVC()
svm.fit(X_train_scaled, y_train_encoded)
y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test_encoded, y_pred_svm))

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train_encoded)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test_encoded, y_pred_knn))

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_gnb))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred_log))
