import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 手动划分训练集和测试集
X_train, X_test = X[:300], X[300:]
y_train, y_test = y[:300], y[300:]

# 特征缩放
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std


# 逻辑回归模型
class LogisticRegression:
    def __init__(self, lr=0.2, num_iter=500, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.losses = []  # 添加一个列表来记录损失值

    def decision_boundary(self, X):
        return (-self.theta[0] - self.theta[1] * X) / self.theta[2]

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # 初始化权重
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            self.losses.append(loss)  # 在每一步中记录损失值

            if self.verbose == True and i % 10000 == 0:
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


# # 感知机模型
# class Perceptron:
#     def __init__(self, learning_rate=0.01, n_iters=1000):
#         self.lr = learning_rate
#         self.n_iters = n_iters
#         self.activation_func = self._unit_step_func
#         self.weights = None
#         self.bias = None
#         self.losses = []
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#
#         # 初始化参数
#         self.weights = np.zeros(n_features)
#         self.bias = 0
#
#         y_ = np.array([1 if i > 0 else 0 for i in y])
#
#         for _ in range(self.n_iters):
#             loss = 0
#             for idx, x_i in enumerate(X):
#                 linear_output = np.dot(x_i, self.weights) + self.bias
#                 y_predicted = self.activation_func(linear_output)
#
#                 # 更新权重和偏置
#                 update = self.lr * (y_[idx] - y_predicted)
#                 self.weights += update * x_i
#                 self.bias += update
#
#                 loss += update ** 2  # 计算损失值
#             self.losses.append(loss)  # 在每一步中记录损失值
#
#     def predict(self, X):
#         linear_output = np.dot(X, self.weights) + self.bias
#         y_predicted = self.activation_func(linear_output)
#         return y_predicted
#
#     def _unit_step_func(self, x):
#         return np.where(x >= 0, 1, 0)


# 训练逻辑回归模型
model = LogisticRegression(lr=0.2, num_iter=500)
model.fit(X_train, y_train)

# # 训练感知机模型
# p = Perceptron(learning_rate=0.2, n_iters=500)
# p.fit(X_train, y_train)

# 预测
preds = model.predict(X_test)
# p_preds = p.predict(X_test)

# 计算准确率
accuracy = (preds == y_test).mean()
# p_accuracy = (p_preds == y_test).mean()

print("LR accuracy:", accuracy)
# print("Perceptron accuracy:", p_accuracy)
# 数据可视化
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='b', label='Not Purchased')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='r', label='Purchased')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# 数据可视化
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='b', label='Not Purchased')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='r', label='Purchased')

# 绘制决策边界
x_values = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
y_values = model.decision_boundary(x_values)
plt.plot(x_values, y_values, color='g', label='Decision Boundary')

plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# 绘制逻辑回归模型的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(model.losses)), model.losses, label='Logistic Regression Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # 绘制感知机模型的损失曲线
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(p.losses)), p.losses, label='Perceptron Loss')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
