# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:48:58 2021

@author: manis
"""
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

# Step 2: Load Datasets
train_data = pd.read_csv("/content/titanic_train_dataset.csv")
test_data = pd.read_csv("/content/Test_titanic_dataset.csv")

# Step 3: Handle Missing Values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Step 4: Feature Scaling and Encoding
scaler = StandardScaler()

# Numerical features
num_features = ['Age', 'Fare']
train_data[num_features] = scaler.fit_transform(train_data[num_features])
test_data[num_features] = scaler.transform(test_data[num_features])

# Categorical features
cat_features = ['Sex', 'Embarked']
encoders = {}

# Encode categorical features using consistent encoders
for feature in cat_features:
    encoder = LabelEncoder()
    combined_values = pd.concat([train_data[feature], test_data[feature]]).unique()
    encoder.fit(combined_values)
    train_data[feature] = encoder.transform(train_data[feature])
    test_data[feature] = encoder.transform(test_data[feature])
    encoders[feature] = encoder

# Step 5: Feature Selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']
X_test_final = test_data[features]

# Step 6: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate Model
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

train_report = classification_report(y_train, y_train_pred)
val_report = classification_report(y_val, y_val_pred)

train_confusion = confusion_matrix(y_train, y_train_pred)
val_confusion = confusion_matrix(y_val, y_val_pred)

# Streamlit App Definition
def main():
    st.title("Titanic Survival Prediction")

    # Dataset preview
    st.subheader("Training Dataset Preview")
    st.write(train_data.head())

    # Model Performance
    st.subheader("Model Performance")
    st.write("Training Set Performance")
    st.write(f"Accuracy: {train_accuracy:.2f}")
    st.text("Classification Report")
    st.text(train_report)
    st.text("Confusion Matrix")
    st.text(np.array2string(train_confusion))

    st.write("Validation Set Performance")
    st.write(f"Accuracy: {val_accuracy:.2f}")
    st.text("Classification Report")
    st.text(val_report)
    st.text("Confusion Matrix")
    st.text(np.array2string(val_confusion))

    # User Input for Prediction
    st.subheader("Predict Titanic Survival")
    Pclass = st.selectbox("Pclass", options=[1, 2, 3])
    Sex = st.selectbox("Sex", options=['male', 'female'])
    Age = st.slider("Age", 0, 80, 25)
    SibSp = st.number_input("Siblings/Spouses Aboard", value=0)
    Parch = st.number_input("Parents/Children Aboard", value=0)
    Fare = st.number_input("Fare", value=0.0)
    Embarked = st.selectbox("Embarked", options=['C', 'Q', 'S'])

    # Encode user input using stored encoders
    Sex = encoders['Sex'].transform([Sex])[0]
    Embarked = encoders['Embarked'].transform([Embarked])[0]
    scaled_features = scaler.transform([[Age, Fare]])[0]

    # Make Prediction
    if st.button("Predict"):
        prediction = model.predict([[Pclass, Sex, scaled_features[0], SibSp, Parch, scaled_features[1], Embarked]])[0]
        result = "Survived" if prediction == 1 else "Did Not Survive"
        st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
