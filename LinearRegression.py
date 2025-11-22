import numpy as np
import logging
from scipy import sparse

# Configure logging for training progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LinearRegression:
    """
    Advanced Linear Regression supporting:
    - Mini-batch Gradient Descent
    - L1 / L2 Regularization
    - Multi-output regression
    - Feature normalization (supports sparse matrices without dense conversion)
    - Convergence check
    - Partial loss calculation (sample fraction)
    - Logging
    """

    def __init__(self, learning_rate: float = 0.01,
                 n_iters: int = 1000,
                 tol: float = 1e-6,
                 normalize: bool = True,
                 loss_every: int = 10,
                 sample_fraction: float = 1.0,
                 batch_size: int = None,
                 regularization: str = None,  # None, 'l1', 'l2'
                 alpha: float = 0.01,
                 random_state: int = None):
        """
        Initialize the Linear Regression model with hyperparameters.
        """
        if not (0 < sample_fraction <= 1):
            raise ValueError("sample_fraction must be between 0 and 1.")
        if regularization not in (None, 'l1', 'l2'):
            raise ValueError("regularization must be None, 'l1', or 'l2'.")

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tol = tol
        self.normalize = normalize
        self.loss_every = loss_every
        self.sample_fraction = sample_fraction
        self.batch_size = batch_size
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state

        # Model parameters
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.history_: list[float] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

        if random_state is not None:
            np.random.seed(random_state)

    def _reshape_input(self, X):
        """
        Ensure X is in the correct shape (2D). Sparse matrices are left unchanged.
        """
        if sparse.issparse(X):
            return X
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _normalize(self, X):
        """
        Normalize features to zero mean and unit variance.
        Supports sparse matrices without converting to dense.
        """
        if not self.normalize:
            return X

        if sparse.issparse(X):
            # Compute mean and std efficiently on sparse data
            self.mean_ = np.array(X.mean(axis=0)).ravel()
            self.std_ = np.sqrt(np.array(X.multiply(X).mean(axis=0)).ravel() - self.mean_ ** 2)
            self.std_[self.std_ == 0] = 1

            # Centering and scaling directly on sparse matrix
            X = X - self.mean_
            X = X.multiply(1 / self.std_)
            return X
        else:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1
            return (X - self.mean_) / self.std_

    def _normalize_predict(self, X):
        """
        Apply the same normalization to new data (for prediction)
        """
        if not self.normalize or self.mean_ is None or self.std_ is None:
            return X

        if sparse.issparse(X):
            X = X - self.mean_
            X = X.multiply(1 / self.std_)
            return X
        else:
            return (X - self.mean_) / self.std_

    def _calculate_loss(self, X, y):
        """
        Compute mean squared error loss, with optional regularization.
        Supports sample_fraction for faster calculation on large datasets.
        """
        m = X.shape[0]

        # Sample subset of data if sample_fraction < 1.0
        if self.sample_fraction < 1.0:
            sample_size = max(1, int(m * self.sample_fraction))
            idx = np.random.choice(m, size=sample_size, replace=False)
            X_sample = X[idx] if not sparse.issparse(X) else X[idx, :]
            y_sample = y[idx]
        else:
            X_sample = X
            y_sample = y

        # Compute predictions
        y_pred = X_sample.dot(self.coef_) + self.intercept_ if sparse.issparse(X_sample) else X_sample @ self.coef_ + self.intercept_
        loss = np.mean((y_pred - y_sample) ** 2)

        # Add regularization term if specified
        if self.regularization == 'l2':
            loss += self.alpha * np.sum(self.coef_ ** 2)
        elif self.regularization == 'l1':
            loss += self.alpha * np.sum(np.abs(self.coef_))
        return float(loss)

    def _get_batch(self, X, y):
        """
        Yield mini-batches of data for Mini-batch Gradient Descent.
        """
        m = X.shape[0]
        if self.batch_size is None or self.batch_size >= m:
            yield X, y
        else:
            idx = np.arange(m)
            np.random.shuffle(idx)
            for start in range(0, m, self.batch_size):
                end = min(start + self.batch_size, m)
                batch_idx = idx[start:end]
                yield (X[batch_idx] if not sparse.issparse(X) else X[batch_idx, :], y[batch_idx])

    def fit(self, X, y):
        """
        Fit the Linear Regression model using (Mini-batch) Gradient Descent.
        """
        X = self._reshape_input(X)
        X = self._normalize(X)
        y = np.array(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        m, n = X.shape
        n_outputs = y.shape[1]

        # Initialize coefficients and intercepts
        self.coef_ = np.zeros((n, n_outputs), dtype=np.float32)
        self.intercept_ = np.zeros((1, n_outputs), dtype=np.float32)

        # Training loop
        for i in range(self.n_iters):
            for X_batch, y_batch in self._get_batch(X, y):
                # Compute predictions
                y_pred = X_batch.dot(self.coef_) + self.intercept_ if sparse.issparse(X_batch) else X_batch @ self.coef_ + self.intercept_
                error = y_pred - y_batch

                # Compute gradients
                dW = (2 / X_batch.shape[0]) * (X_batch.T.dot(error) if sparse.issparse(X_batch) else X_batch.T @ error)
                db = (2 / X_batch.shape[0]) * error.sum(axis=0, keepdims=True)

                # Apply regularization
                if self.regularization == 'l2':
                    dW += 2 * self.alpha * self.coef_
                elif self.regularization == 'l1':
                    dW += self.alpha * np.sign(self.coef_)

                # Update coefficients
                self.coef_ -= self.learning_rate * dW
                self.intercept_ -= self.learning_rate * db

            # Compute loss periodically
            if i % self.loss_every == 0:
                loss = self._calculate_loss(X, y)
                self.history_.append(loss)
                # Check convergence
                if len(self.history_) > 1 and abs(self.history_[-2] - self.history_[-1]) < self.tol:
                    logging.info(f"Converged at iteration {i} with loss {loss:.6f}")
                    break
        return self

    def predict(self, X):
        """
        Predict target values for new data.
        """
        X = self._reshape_input(X)
        X = self._normalize_predict(X)
        return X.dot(self.coef_) + self.intercept_ if sparse.issparse(X) else X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Compute R^2 score of the model on given data.
        """
        X = self._reshape_input(X)
        y = np.array(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
        return 1 - ss_res / ss_tot

    def get_params(self, deep: bool = True) -> dict:
        """
        Return model hyperparameters (like sklearn interface)
        """
        return {
            "learning_rate": self.learning_rate,
            "n_iters": self.n_iters,
            "tol": self.tol,
            "normalize": self.normalize,
            "loss_every": self.loss_every,
            "sample_fraction": self.sample_fraction,
            "batch_size": self.batch_size,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "random_state": self.random_state
        }

    def set_params(self, **params) -> "LinearRegression":
        """
        Set model hyperparameters (like sklearn interface)
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self
