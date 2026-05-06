import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Training.csv")

# Remove unwanted column if exists
if "Unnamed: 133" in df.columns:
    df = df.drop("Unnamed: 133", axis=1)

# Features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel='linear')

print("Training model...")
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
pickle.dump(model, open("svc.pkl", "wb"))

print("New model saved as svc.pkl")