import numpy as np
import linearRegression
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)

size = (np.random.rand(100, 1) * 2 + 0.3) * 100  # in square meters
price = size * 1.1 + np.random.rand(100, 1)  # in 10k rmb
ax[0, 0].scatter(size, price, c='b')


lr = linearRegression.LinearRegression()
lr.set_x(size)
lr.set_y(price)
lr_1 = size * lr.theta[1][0] + lr.theta[0][0]
ax[0, 0].plot(size, lr_1, 'r')


lr.z_score_normalization()
ax[0, 1].scatter(lr.x, lr.y, c='b')
lr_2 = lr.x * lr.theta[1][0] + lr.theta[0][0]
ax[0, 1].plot(lr.x, lr_1, 'r')


costs = lr.gradient_descent()
a = np.array([i for i in range(3000)])
ax[1, 0].plot(a, costs, marker=".")


ax[1, 1].scatter(size, price, c='b')
lr_3 = size * lr.theta[1][0] + lr.theta[0][0]
ax[1, 1].plot(size, lr_2, c='r')

plt.show()
