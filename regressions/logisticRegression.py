import numpy as np
from typing import Tuple

from numpy import ndarray


class LogisticRegression:
    """
    After initiating an instance, the method 'set_x()' and 'set_y()'
    must be called in order to make the class correctly function.
    """
    theta: np.ndarray
    alpha: float  # learning rate.
    m: int  # size of training set.
    x: np.ndarray  # features or input variables; should be of size '(m, 1)'.
    y: np.ndarray  # target variables or output variables; should be of size '(m, 1)'.
    sigmoid: np.ndarray  # a vector for the computed sigmoid function.

    def __init__(self):
        self.theta = np.random.randn()
        self.alpha = 0.1

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

    def set_x(self, x) -> None:
        if self._update_m(len(x)):
            self.x = np.c_[np.ones((self.m,)), x]

    def set_y(self, y) -> None:
        if self._update_m(len(y)):
            self.y = y

    def get_costs(self) -> ndarray:
        self.get_sigmoid()
        return np.sum(-1 * self.y * np.log(self.sigmoid) - (1 - self.y) * np.log(1 - self.sigmoid))

    def get_sigmoid(self) -> None:
        z = self.x.dot(self.theta)
        self.sigmoid = 1 / (1 + np.exp(-1 * z))

    def gradient_descent(self, iteration=3000) -> None:
        self.get_sigmoid()
        for i in range(iteration):
            self.theta = self.theta - (1 / self.m) * (self.x.T.dot(self.sigmoid - self.y))

    def predict(self, x: float) -> float:
        return 1 / (1 + x * self.theta[1][0] + self.theta[0][0])
