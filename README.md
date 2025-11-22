# Build Advanced Linear Regression From Scratch

This repository contains a professional implementation of **Linear Regression from scratch** in Python with advanced features for large-scale and sparse data.  

Unlike traditional Linear Regression implementations in libraries like **scikit-learn**, which often rely on the **Normal Equation**, this implementation uses **Stochastic / Mini-batch Gradient Descent (SGD)** with **early stopping**, normalization, and regularization.  

---

## Overview

Linear Regression predicts continuous target values.  
We aim to find parameters `theta` that minimize the **Mean Squared Error (MSE)**.

**Mean Squared Error (MSE):**

Where:
- `m` = number of samples  
- `X_i` = feature vector of sample i  
- `y_i` = target value of sample i  
- `theta` = parameter vector  

---

## How to Calculate Theta

- **Normal Equation (closed-form, like scikit-learn):**

> Pros: one-step calculation, exact solution.  
> Cons: very slow or impossible for large datasets, cannot handle sparse efficiently.

- **Gradient Descent (iterative):**
Where `alpha` is the learning rate.  

- **Mini-batch / Stochastic Gradient Descent (SGD):**

Where:
- `B` = mini-batch of samples  
- Updates `theta` iteratively  
- Supports early stopping for faster convergence  

---

## Features

- Mini-batch Gradient Descent for large datasets  
- Early stopping to speed up training  
- Normalization of dense and sparse features  
- L1 (Lasso) and L2 (Ridge) regularization  
- Multi-output regression (predict multiple targets)  
- Partial loss calculation for very large datasets  
- Sklearn-like API (`fit`, `predict`, `score`, `get_params`, `set_params`)  
- Logging for training progress  

---

## Advantages of SGD over Normal Equation

1. Works efficiently with **large datasets**  
2. Supports **sparse data** without converting to dense  
3. Early stopping prevents unnecessary iterations  
4. Mini-batch updates can leverage **GPU acceleration**  
5. Supports **regularization** natively  

---

## Limitations / Disadvantages

1. Requires **hyperparameter tuning** (learning rate, batch size, regularization strength)  
2. Convergence depends on **initialization** and learning rate  
3. Iterative updates can be slower than Normal Equation for **small datasets**  
4. Gradient descent may not reach exact minimum, only approximate  

---

## SGD vs Normal Equation Diagram

![SGD vs Normal Equation](docs/sgd_vs_normal_equation.png)

> Diagram shows the difference:  
> - **Normal Equation:** one-step exact solution  
> - **SGD:** iterative updates, faster for big data  

---

## Loss Convergence Example

While training, the **loss decreases over iterations**:

```python
import matplotlib.pyplot as plt

plt.plot(model.history_)
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Loss Convergence During Training")
plt.show()


