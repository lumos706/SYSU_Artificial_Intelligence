import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def plot_data_and_decision_boundary(X, y, svm):
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

    # 获取坐标轴的边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 生成网格点坐标
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 对网格点进行预测
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, cmap='Dark2', alpha=0.2)

    # 设置坐标轴标签和标题
    plt.xlabel('Iris Feature 3')
    plt.ylabel('Iris Feature 4')
    plt.title('Data and Decision Boundary')

    # 显示图形
    plt.show()
    plt.savefig('../data_and_decision_boundary.png')


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            loss = sum(
                max(0, 1 - y_[idx] * (np.dot(x_i, self.w) - self.b)) + self.lambda_param * np.linalg.norm(self.w) ** 2
                for idx, x_i in enumerate(X))
            self.losses.append(loss)

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def plot_loss_curve(self):
        plt.plot(self.losses)
        plt.title('Loss curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
        plt.savefig('../loss_curve.png')


def main():
    # 加载数据集
    iris = datasets.load_iris()
    X = iris.data[:100, 2:4]  # 只取前100条数据（Setosa和Versicolor），并只取第三和第四个特征
    y = iris.target[:100]  # 只取前100条数据的label

    # 将标签中的0替换为-1
    y[y == 0] = -1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练SVM分类器
    svm = LinearSVM()
    svm.fit(X_train, y_train)

    # 预测测试集
    predictions = svm.predict(X_test)

    # 计算准确率
    print(f"SVM分类器的准确率：{accuracy_score(y_test, predictions)*100}%")
    plot_data_and_decision_boundary(X, y, svm)
    svm.plot_loss_curve()


if __name__ == "__main__":
    main()