# Linear Regression - Custom Implementation in a Jupyter Notebook
This repository contains a custom implementation of a linear regression algorithm, built from scratch without using any machine learning libraries like scikit-learn. The goal of this project is to gain a deep understanding of the logic and mathematics that underline linear regression through practical implementation.

## What is Linear Regression?
Linear regression is a fundamental algorithm in machine learning that models the relationship between a dependent variable and one or more independent variables. The case of one explanatory variable is called simple linear regression, for more than one, the process is called multiple linear regression.

### Files
- Linear_Regression_Notebook.ipynb: This Jupyter Notebook contains the complete workflow for training a linear regression model, including data preparation, model training using both batch and stochastic gradient descent, and visualizations of results and residuals.
- boston_train.csv: The training dataset used to train the linear regression model.
- boston_test.csv: The testing dataset used for making predictions and visualizing how the model performs on unseen data.
### How it Works?
The Linear_Regression_Notebook.ipynb does the following:

- Imports Necessary Libraries: Starts by importing essential Python libraries including pandas for data manipulation and numpy for numerical operations.
- Prepares Data: Reads in the boston_train.csv file, separates the target variable and features, normalizes the features, and adds a column of ones to account for the bias term in linear regression.
- Initializes Parameters: The weights for the linear regression model are initialized randomly. The learning rate (alpha) and regularization parameter (beta) are set.
- Gradient Descent Implementation: It performs batch gradient descent on the entire dataset and stochastic gradient descent, updating weights based on the gradient of the cost function.
- Comparison and Analysis: Compares the effectiveness of batch vs. stochastic gradient descent methods in terms of convergence speed and prediction accuracy.
### Running the Notebook
To run this notebook, you will need an environment capable of executing Jupyter Notebooks with Python installed, along with the pandas and numpy libraries. It is recommended to use Anaconda as it simplifies package management and deployment of the Jupyter environment.

## Educational Purpose
This implementation is intended for educational purposes, providing hands-on experience with the fundamentals of linear regression and gradient descent methods. It is not optimized for large datasets or for use in production environments.
