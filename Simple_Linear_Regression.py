import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None  # Coefficient (slope) of the line
        self.intercept_ = None  # Intercept of the line

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        :param X: Training data (independent variable)
        :param y: Target values (dependent variable)
        """
        # Calculate the mean of X and y
        mean_x = np.mean(X)
        mean_y = np.mean(y)

        # Calculate the total number of data points
        n = len(X)

        # Calculate the slope (coefficient)
        numerator = np.sum((X - mean_x) * (y - mean_y))
        denominator = np.sum((X - mean_x) ** 2)
        self.coef_ = numerator / denominator

        # Calculate the intercept
        self.intercept_ = mean_y - self.coef_ * mean_x

    def predict(self, X):
        """
        Make predictions for new data.

        :param X: New data points
        :return: Predicted values
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model has not been trained yet. Please use the 'fit' method to train the model.")
        return self.coef_ * X + self.intercept_

# Sample data for demonstration
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# Create a SimpleLinearRegression model
model = SimpleLinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
new_data_point = 6
prediction = model.predict(new_data_point)
print(f"Prediction for x={new_data_point}: {prediction}")
