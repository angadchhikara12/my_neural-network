# Neural Network for simple addition: Answer to Science Question 
# *How similar is Neural Network to Human Brain?*

## Features
- **Trainable Neural Network**: The model is trained to perform addition on two input numbers.
- **Interactive Testing**: Users can input two numbers, and the AI predicts the sum with a confidence score.
- **Mimicking Human Learning**: The project highlights how neural networks "learn" from data without being explicitly programmed for the task.

---

## How it works
1. **Training Data**: The network is trained with random pairs of numbers and their sums.
2. **Neural Network Architecture**:
   - **Input Layer**: Takes two input numbers.
   - **Hidden Layers**: Two layers with 16 neurons each, using ReLU activation.
   - **Output Layer**: Predicts a single value (the sum of the inputs).
3. **Confidence Calculation**: The confidence of each prediction is calculated based on the error margin.
4. **User Interaction**: Users input numbers, and the AI predicts their sum while displaying a confidence score.

---

## Technologies Used
 - **Python:** Primary programming language
 - **Tensorflow/Keras:** For building and training the Neural Network
 - **Numpy:** For data generation and numerical operators.
---
## Prerequisites
1. Install Python 3.10.11
2. Install the required Python Libraries
    ```bash
   pip install tensorflow numpy
    ```
---
## Usage
### 1. Clone Repository
```bash
  git clone https://github.com/@angadchhikara12/my_nerual-network.git
  cd my_nerual-network
 ```
### 2. Run the Script
```bash
   python network.py
```
##### or
```bash
   python -m network
```

### 3. Test the Neural Network
 - Input 2 numbers when prompted (eg. 5 3)
 - The Neural Network will:
   - Predict the sum.
   - Display the true answer.
   - Show a confidence percentage
---
