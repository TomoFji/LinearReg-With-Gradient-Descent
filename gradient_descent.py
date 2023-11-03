import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Feature Scaling (Standardization)
from sklearn.preprocessing import StandardScaler

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, maxiter=1000, random_init=True, features=None):
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.random_init = random_init
        self.features = features
        self.weights = None
        self.bias = None

    # If random_init is True, weights and bias will be initialized with random values between 0 and 1; otherwise, they will be initialized to 0. Default is random initialization.
    def initialize_parameters(self):
        if self.features is not None:
            if self.random_init:
                self.weights = np.random.rand(self.features)
                self.bias = np.random.rand()
            else:
                self.weights = np.zeros(self.features)
                self.bias = 0
        else:
            self.weights = None
            self.bias = None

    # Trains the model using gradient descent
    def fit(self, x, y):
        samples, features = x.shape

        # Initialize weights and bias if not done already
        if self.weights is None:
            self.initialize_parameters()

        for i in range(self.maxiter):
            y_pred = np.dot(x, self.weights) + self.bias
            dw = (1 / samples) * np.dot(x.T, (y_pred - y))
            db = (1 / samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Returns predictions after training the model
    def predict(self, x):
        if self.weights is not None:
            return np.dot(x, self.weights) + self.bias
        else:
            raise ValueError("Model has not been fitted. Call fit method to train the model")

if __name__ == '__main__':
    df = pd.read_csv('market_data.csv')

    # Extract features and target
    y = df['Sale'].to_numpy()
    x = df[['Price', 'Discount']].to_numpy()

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Select a specific range of the original data
    price_range_indices = (x[:, 0] >= 0) & (x[:, 0] <= 1)  # Adjust the range as needed
    x_range = x[price_range_indices]
    y_range = y[price_range_indices]

    # Model creation
    model = LinearRegressionGD(features=x_range.shape[1])
    model.fit(x_range, y_range)

    plt.scatter(x_range[:, 0], y_range, color='blue', label='Data Points')
    plt.scatter(x_range[:, 1], y_range, color='purple', label='Data Points')

    # Generate x_values and corresponding y_predictions for the regression line
    x_values = np.linspace(-2, 2, 100)  # Adjust the range to match the selected data range
    x_for_prediction = np.column_stack((x_values, x_values))  # Match the shape of x

    y_pred = model.predict(x_for_prediction)

    # Create a line plot to visualize the regression line
    plt.plot(x_values, y_pred, color='red', label='Regression Line')

    plt.title('Linear Regression Model')
    plt.xlabel('Price and Discount')
    plt.ylabel('Sale')
    plt.legend()
    plt.show()