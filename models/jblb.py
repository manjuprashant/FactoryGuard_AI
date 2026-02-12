import joblib

# Load the model
model = joblib.load("models/failure_model.joblib")

# Make a prediction
sample_input = [[0.5, 0.05, 70, 2, 30, 1]]  # Example feature values
prediction = model.predict(sample_input)

print(prediction)
