import numpy as np
import logging
from scipy import sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LinearRegression:
    """
    Professional, production-ready Linear Regression from scratch.
    Full scikit-learn compatibility + sparse support + correct L1/L2 + early stopping.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iters: int = 1000,
        tol: float = 1e-6,
        normalize: bool = True,
        loss_every: int = 10,
        sample_fraction: float = 1.0,
        batch_size: int | None = None,
        regularization: str | None = None,  # None, 'l1', 'l2'
        alpha: float = 0.01,
        random_state: int | None = None,
        early_stopping: bool = True,
        patience: int = 10,
    ):
        if not (0 < sample_fraction <= 1):
            raise ValueError("sample_fraction must be in (0, 1]")
        if regularization not in (None, "l1", "l2"):
            raise ValueError("regularization must be None, 'l1', or 'l2'")

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
        self.early_stopping = early_stopping
        self.patience = patience

        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.history_ = []
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self._is_fitted = False

        if random_state is not None:
            np.random.seed(random_state)

    def _reshape_X(self, X):
        if sparse.issparse(X):
            return X
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if X.size == 1 else X.reshape(1, -1)
        return X

    def _reshape_y(self, y):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def _normalize(self, X):
        """Sparse-safe normalization without densifying."""
        if not self.normalize:
            return X

        if sparse.issparse(X):
            self.mean_ = np.ravel(X.mean(axis=0))
            X_centered = X.copy()
            X_centered.data -= np.take(self.mean_, X_centered.indices)
            var = np.ravel(X.multiply(X).mean(axis=0)) - self.mean_ ** 2
            self.std_ = np.sqrt(var + 1e-12)
            self.std_[self.std_ == 0] = 1.0
            scale = sparse.diags(1.0 / self.std_)
            return scale @ X_centered
        else:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-12
            self.std_[self.std_ == 0] = 1.0
            return (X - self.mean_) / self.std_

    def _normalize_predict(self, X):
        if not self.normalize or self.mean_ is None:
            return X
        if sparse.issparse(X):
            X_centered = X.copy()
            X_centered.data -= np.take(self.mean_, X_centered.indices)
            scale = sparse.diags(1.0 / self.std_)
            return scale @ X_centered
        return (X - self.mean_) / self.std_

    def _calculate_loss(self, X, y):
        m = X.shape[0]
        scale = 1.0 / self.sample_fraction if self.sample_fraction < 1.0 else 1.0
        idx = slice(None) if self.sample_fraction >= 1.0 else np.random.choice(
            m, max(1, int(m * self.sample_fraction)), replace=False
        )
        X_s, y_s = X[idx], y[idx]

        y_pred = X_s @ self.coef_ + self.intercept_
        mse = np.mean((y_pred - y_s) ** 2)

        reg = 0.0
        if self.regularization == "l2":
            reg = self.alpha * np.sum(self.coef_ ** 2)
        elif self.regularization == "l1":
            reg = self.alpha * np.sum(np.abs(self.coef_))

        return float(mse + scale * reg)

    def _get_batches(self, X, y):
        m = X.shape[0]
        if self.batch_size is None or self.batch_size >= m:
            yield X, y
            return
        indices = np.random.permutation(m)
        for i in range(0, m, self.batch_size):
            batch = indices[i:i + self.batch_size]
            yield (X[batch] if not sparse.issparse(X) else X[batch, :], y[batch])

    def fit(self, X, y):
        X = self._reshape_X(X)
        y = self._reshape_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        X = self._normalize(X)
        n_features, n_outputs = X.shape[1], y.shape[1]

        self.n_features_in_ = n_features
        self.coef_ = np.zeros((n_features, n_outputs), dtype=np.float64)
        self.intercept_ = np.zeros(n_outputs, dtype=np.float64)  # 1D intercept

        no_improve_count = 0
        best_loss = np.inf

        for it in range(self.n_iters):
            for Xb, yb in self._get_batches(X, y):
                y_pred = Xb @ self.coef_ + self.intercept_
                error = y_pred - yb

                grad_w = (2 / len(yb)) * (Xb.T @ error)
                grad_b = (2 / len(yb)) * error.sum(axis=0)

                if self.regularization == "l2":
                    grad_w += 2 * self.alpha * self.coef_

                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b

                if self.regularization == "l1":
                    threshold = self.learning_rate * self.alpha
                    self.coef_ = np.sign(self.coef_) * np.maximum(np.abs(self.coef_) - threshold, 0.0)

            if it % self.loss_every == 0:
                loss = self._calculate_loss(X, y)
                self.history_.append(loss)
                logging.info(f"Iter {it} | Loss: {loss:.6f}")

                if self.early_stopping:
                    if loss + self.tol < best_loss:
                        best_loss = loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    if no_improve_count >= self.patience:
                        logging.info(f"Early stopping at iteration {it} | Loss: {loss:.6f}")
                        break

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call 'fit' first.")
        X = self._reshape_X(X)
        X = self._normalize_predict(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")
        X = self._reshape_X(X)
        y = self._reshape_y(y)
        X = self._normalize_predict(X)
        y_pred = self.predict(X)

        r2_per_output = []
        for i in range(y.shape[1]):
            ss_res = np.sum((y[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y[:, i] - y[:, i].mean()) ** 2)
            r2_per_output.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return float(np.mean(r2_per_output))

    def get_params(self, deep=True):
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
            "random_state": self.random_state,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")
            setattr(self, key, value)
        if params.get("random_state") is not None:
            np.random.seed(params["random_state"])
        return self
