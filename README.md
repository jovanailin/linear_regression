# Linear Regression - Custom Implementation

This repository contains an implementation of a linear regression algorithm, built from scratch without using any machine learning libraries like scikit-learn. The goal of this project is to get a deep understanding of the logic and mathematics that underlie linear regression.

## What is Linear Regression?

Linear regression is a fundamental algorithm in machine learning that models the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.

### Files

`linear_regression.py`: This is the main Python script where the linear regression algorithm is implemented from scratch.

`boston.csv`: This is the training dataset used to train the linear regression model.

`boston_novi.csv`: This is the testing dataset used to validate the model and predict the output.

### How it works?

The `linear_regression.py` script does the following:

1. Imports necessary libraries: It starts by importing the necessary libraries (pandas and numpy), and setting some parameters.

2. Prepares data: The script reads in a CSV file named 'boston.csv', then separates the target variable (y) and the features (X). The features are normalized, and a column of ones is added to the feature set to account for the bias term in the linear regression.

3. Initializes parameters: The weights for the linear regression model are initialized randomly. The learning rate (alpha) and regularization parameter (beta) are set.

4. Gradient Descent: It performs one iteration of batch gradient descent on the entire dataset, updating the weights based on the gradient of the cost function.

5. Stochastic Gradient Descent: It then shuffles the dataset and performs stochastic gradient descent, updating the weights one example at a time.

6. Predicts: Finally, it reads in a new CSV file named 'boston_novi.csv', normalizes it using the same parameters as the training set, and makes predictions based on the final weights learned from the gradient descent.

### Running the Code

To run the code, you will need to have Python installed on your computer along with the pandas and numpy libraries. Once those are installed, you can run the `linear_regression.py` script using any Python IDE or through the command line.

Please note that this implementation is educational and not optimized for large datasets or for use in production.

