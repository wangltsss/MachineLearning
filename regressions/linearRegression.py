import numpy as np
from typing import Tuple


class LinearRegression:
    theta: np.ndarray
    alpha: float  # learning rate.
    m: int  # size of training set.
    x: np.ndarray  # features or input variables; should be of size '(m, 1)'.
    y: np.ndarray  # target variables or output variables; should be of size '(m, 1)'.

    def __init__(self):
        self.theta = np.random.randn(2, 1)
        self.alpha = 0.1

    def set_x(self, x: np.ndarray) -> None:
        if self._update_m(len(x)):
            self.x = x

    def set_y(self, y: np.ndarray) -> None:
        if self._update_m(len(y)):
            self.y = y

    def set_learning_rate(self, alpha: float) -> None:
        self.alpha = alpha

    def _update_m(self, m: int) -> bool:
        """
        check if sizes of two variables matches the size of training set.
        :param m: size of training set variable.
        :return: True if current size of training set is not set,
         or the new size of training set matches the existing one;
         False the otherwise.
        """
        try:
            if self.m != m:
                return False
            return True
        except AttributeError:
            self.m = m
            return True
        except Exception as err_msg:
            print(err_msg)
            return False

    def get_squared_error(self) -> float:
        """
        the cost function of current 'w' and 'b' or 'theta.
        :return: the squared error or the cost.
        """
        return (1 / 2 * self.m) * np.sum(np.square((self.x.dot(self.theta)) - self.y))

    def gradient_descent(self, iteration=3000) -> np.ndarray:
        self.x = np.c_[np.ones((self.m, )), self.x]
        thetas = np.zeros((iteration, 2))
        costs = np.zeros(iteration)
        for i in range(iteration):
            self.theta = self.theta - (1 / self.m) * self.alpha * (self.x.T.dot((self.x.dot(self.theta)) - self.y))
            thetas[i, :] = self.theta.T
            costs[i] = self.get_squared_error()
        return costs

    def z_score_normalization(self) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.mean(self.x)
        sig = np.std(self.x)
        self.x = (self.x - mu) / sig
        return mu, sig

    def predict(self, x: float) -> float:
        return x * self.theta[1][0] + self.theta[0][0]


















