# train_model.py
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "app/model.pkl")  # Saves inside app/ folder
