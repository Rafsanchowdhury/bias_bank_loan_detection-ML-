import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
def load_data(file_path):
    df = pd.read_excel(file_path)
    print("Dataset Loaded Successfully")
    print(df.head())
    return df

# Preprocess the dataset
def preprocess_data(df):
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Loan Status'] = df['Loan Status'].map({'Approved': 1, 'Declined': 0})
    print("Data Preprocessing Completed")
    return df

# Feature Selection and Splitting
def feature_selection(df):
    X = df[['Gender', 'Income', 'Credit Score']]
    y = df['Loan Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Feature Selection and Train-Test Split Done")
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model Training Completed")
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

# Analyze Bias
def analyze_bias(df, model):
    male_data = df[df['Gender'] == 0]
    female_data = df[df['Gender'] == 1]

    male_approval_rate = model.predict(male_data[['Gender', 'Income', 'Credit Score']]).mean()
    female_approval_rate = model.predict(female_data[['Gender', 'Income', 'Credit Score']]).mean()

    print("Bias Analysis Results:")
    print(f"Male Approval Rate: {male_approval_rate * 100:.2f}%")
    print(f"Female Approval Rate: {female_approval_rate * 100:.2f}%")

# Main Function
def main():
    file_path = "data/Aminas_Bank_Data.xlsx"
    df = load_data(file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = feature_selection(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    analyze_bias(df, model)

if __name__ == "__main__":
    main()
