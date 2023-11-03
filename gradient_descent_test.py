import pytest
import numpy as np
from gradient_descent import LinearRegressionGD 

# Define the fixture to create an instance of the LinearRegressionGD class
@pytest.fixture
def linear_regression_model():
    return LinearRegressionGD(learning_rate=0.01, maxiter=1000, random_init=True, features=2)

# Test initialization of LinearRegressionGD class
def test_linear_regression_initialization(linear_regression_model):
    model = linear_regression_model
    assert model.learning_rate == 0.01
    assert model.maxiter == 1000
    assert model.random_init is True
    assert model.features == 2
    assert model.weights is None
    assert model.bias is None

# Test initialization of parameters
def test_initialize_parameters(linear_regression_model):
    model = linear_regression_model
    model.initialize_parameters()
    assert model.weights is not None
    assert model.bias is not None

# Test model training and prediction
def test_linear_regression_training_and_prediction(linear_regression_model):
    model = linear_regression_model

    # Generate some sample data for testing
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([3.0, 4.0, 5.0])

    # Fit the model to the data
    model.fit(x, y)

    # Test prediction
    x_pred = np.array([[4.0, 5.0]])
    y_pred = model.predict(x_pred)

    # Ensure the model's predictions are close to the expected values
    assert np.allclose(y_pred, np.array([6.0]), atol=0.2)

if __name__ == '__main__':
    pytest.main()