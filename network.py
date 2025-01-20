import keras
from keras._tf_keras.keras.models import load_model
import numpy as np


# Step 1: Generate Training Data
def generate_training_data():
    # Randomly generate numbers and let the network learn addition
    x_train = np.random.randint(0, 100, (1000, 2))  # Two input numbers
    y_train = np.sum(x_train, axis=1)  # Compute their sum as the label
    return x_train.astype(np.float32), y_train.astype(np.float32)


# Step 2: Build Neural Network Model
def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),  # Input layer (2 numbers)
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)  # Output layer (single result)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Step 3: Train the Neural Network
def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)


# Step 4: Calculate Confidence
def calculate_confidence(predicted_value, true_value, threshold=5):
    # Calculate the absolute error
    error = abs(predicted_value - true_value)
    # Confidence is higher for lower errors
    confidence = max(0, 1 - (error / threshold))
    return confidence * 100  # Return as a percentage


# Step 5: Test the Neural Network
def test_model(model):
    for i in range(5):
        # Take input from the user
        print("-----------------------------------------------------------------------------------------------------------------------")
        user_input = input("Enter two numbers to add (e.g., 5 3) or 'quit' to exit: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        try:
            # Parse user input
            num1, num2 = map(float, user_input.split())
            input_data = np.array([[num1, num2]], dtype=np.float32)

            # Predict the result
            prediction = model.predict(input_data)
            predicted_value = prediction[0][0]
            true_value = num1 + num2  # The actual addition result

            # Calculate confidence
            confidence = calculate_confidence(predicted_value, true_value)

            # Display results
            print(f"The AI predicts: {predicted_value:.2f}")
            print(f"True Answer: {true_value:.2f}")
            print(f"Confidence: {confidence:.2f}%\n")
        except ValueError:
            print("Invalid input. Please enter two numbers.")


# Main Program
if __name__ == "__main__":
    # Generate Data
    x_train, y_train = generate_training_data()

    # Build and Train Model
    model = build_model()
    train_model(model, x_train, y_train)

    # Test Model
    test_model(model)
