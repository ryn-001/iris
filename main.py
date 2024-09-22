# Import necessary libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('Iris.csv')
X = data.drop(columns=['Id', 'Species'])  # Remove 'Id' and 'Species' columns for training features
y = data['Species']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model to a file using pickle
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")

# Function to predict flower species based on user input
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Print received inputs for debugging
    print(f"Inputs received - Sepal Length: {sepal_length}, Sepal Width: {sepal_width}, Petal Length: {petal_length}, Petal Width: {petal_width}")

    # Check if inputs are within expected ranges
    if (sepal_length < 4.0 or sepal_length > 8.0 or
        sepal_width < 2.0 or sepal_width > 5.0 or
        petal_length < 1.0 or petal_length > 7.0 or
        petal_width < 0.1 or petal_width > 2.5):
        return "Input values are out of the expected range based on the dataset."

    # Create a DataFrame from the input, using the same column names as the training data
    input_data = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    })

    # Make a prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Load the model from the file
try:
    with open('iris_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please ensure the model is trained and saved.")

# Example: Predict using user inputs
try:
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    # Use the loaded model for prediction
    predicted_species = loaded_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(f"The predicted species of the iris flower is: {predicted_species[0]}")

except ValueError as e:
    print("Please enter valid numerical values.", e)
