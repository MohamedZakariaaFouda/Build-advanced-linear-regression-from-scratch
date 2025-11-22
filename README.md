# Build Advanced Linear Regression From Scratch

This repository contains a professional implementation of **Linear Regression from scratch** in Python with advanced features for large-scale and sparse data.  

Unlike traditional Linear Regression implementations in libraries like **scikit-learn**, which often rely on the **Normal Equation**, this implementation uses **Stochastic / Mini-batch Gradient Descent (SGD)** with **early stopping**, normalization, and regularization.  

---

## Overview

Linear Regression is a supervised learning algorithm used to predict continuous target variables.  

Given a dataset \(X \in \mathbb{R}^{m \times n}\) and target \(y \in \mathbb{R}^{m}\), we aim to learn parameters \(\theta \in \mathbb{R}^{n}\) that minimize the **Mean Squared Error (MSE)**:

\[
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (y_i - X_i \cdot \theta)^2
\]

- **Normal Equation (used in scikit-learn)**:

\[
\theta = (X^T X)^{-1} X^T y
\]

- **Gradient Descent (used here)**:

\[
\theta := \theta - \alpha \nabla_\theta J(\theta)
\]

Where the gradient is:

\[
\nabla_\theta J(\theta) = \frac{2}{m} X^T (X \theta - y)
\]

**Mini-batch / Stochastic Gradient Descent (SGD)**: instead of using all data points for each update, we use a subset (batch) which speeds up training on large datasets:

\[
\theta := \theta - \alpha \frac{2}{|B|} X_B^T (X_B \theta - y_B)
\]

- Supports **multi-output regression**, L1/L2 **regularization**, and **feature normalization**, including **sparse matrices** without converting to dense.  

---

## Features

- **Mini-batch Gradient Descent** for faster training on large datasets.  
- **Early stopping** using `tol` and `loss_every` to terminate training when improvement is small.  
- **Normalization of features**, including sparse matrices efficiently.  
- **Regularization**: L1 (Lasso) and L2 (Ridge) support.  
- **Multi-output regression**: predict multiple target variables at once.  
- **Partial loss calculation** for efficiency on very large datasets.  
- **Sklearn-like API**: `fit`, `predict`, `score`, `get_params`, `set_params`.  
- **Logging** for training progress.

---

## Advantages over Normal Equation

1. Works efficiently on **large datasets** where \(X^T X\) inversion is slow or impossible.  
2. Supports **sparse data** without dense conversion.  
3. **Early stopping** allows faster convergence without iterating unnecessarily.  
4. **Mini-batch gradient descent** can leverage GPU or parallel computation.  
5. Supports **regularization**, which normal equation without modification does not handle directly.

---

## Limitations / Disadvantages

1. Requires **hyperparameter tuning** (learning rate, batch size, regularization).  
2. Convergence depends on **initialization** and learning rate.  
3. Gradient Descent may be slower on very small datasets compared to normal equation.  
4. Requires **multiple iterations** to converge, unlike normal equation which is closed-form.  

---

## Usage Example

```python
import numpy as np
from linear_regression import LinearRegression
from scipy import sparse

# Example sparse data
X = sparse.random(1000, 20, density=0.1, format='csr', random_state=42)
y = np.random.rand(1000)

# Initialize model
model = LinearRegression(
    learning_rate=0.01,
    n_iters=1000,
    batch_size=32,
    normalize=True,
    regularization='l2',
    alpha=0.1,
    loss_every=10
)

# Fit model
model.fit(X, y)

# Predict
preds = model.predict(X)

# R^2 score
print("R^2 score:", model.score(X, y))
