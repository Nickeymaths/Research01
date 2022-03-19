import time

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)


class LinearModel:
    def __init__(self, learning_rate, degree, epoch):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.theta = np.zeros(degree+1)

    def fit(self, x, y):
        """
        Training model with closed form optimization
        :param x: input of training data, ndarray (m, n),
        :param y: output of trianing data, ndarry (m,)
        """
        feature = self.create_feature(x)
        self.theta = np.linalg.pinv(feature.T.dot(feature)).dot(feature.T).dot(y)

    def create_feature(self, x):
        """
        Mapping from input attribute to feature
        :param x: input attributes
        :return: input features
        """
        feature = []
        for i in range(self.degree + 1):
            feature.append(x ** i)
        return np.array(feature).T

    def predict(self, x):
        """
        :param x: ndarray (k,n)
        :return: model prediction on x
        """
        return self.create_feature(x).dot(self.theta)


def loss(y_pred, y):
    return np.sqrt(np.sum((y_pred - y)**2)/len(y))


def create_fake_data(savef):
    v0, g = 20, 5
    X = np.random.rand(100, )*8

    # Generate noise
    eps = np.random.normal(0, 3, (100,))
    Y = v0*X-1/2*g*X**2 + eps
    bayes_error = np.sqrt(np.sum(eps**2)/len(eps))
    data = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    # Save data
    np.savetxt(savef, data)
    return bayes_error


if __name__ == '__main__':
    savef = './data/free_dropping/data.csv'
    bayes_error = create_fake_data(savef)
    # Load data
    data = np.loadtxt(savef)
    X, y = data[:, 0], data[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # View data
    plt.figure(figsize=(8, 8))
    plt.scatter(X, y)
    plt.savefig('./output/free_dropping/data_plot.png')
    # Train experiments
    time_0 = time.time()
    model1 = LinearModel(learning_rate=0.1, degree=1, epoch=10)
    model1.fit(X_train, y_train)
    loss_r01 = loss(model1.predict(X_test), y_test)
    print("Finished time of R01 %.2f", time.time() - time_0)

    time_0 = time.time()
    model2 = LinearModel(learning_rate=0.1, degree=2, epoch=10)
    model2.fit(X_train, y_train)
    loss_r02 = loss(model2.predict(X_test), y_test)
    print("Finished time of R02 %.2f", time.time() - time_0)

    time_0 = time.time()
    model3 = LinearModel(learning_rate=0.1, degree=3, epoch=10)
    model3.fit(X_train, y_train)
    loss_r03 = loss(model3.predict(X_test), y_test)
    print("Finished time of R03 %.2f", time.time() - time_0)
    # Evaluate experiments

    if loss_r01 <= bayes_error or loss_r02 <= bayes_error or loss_r03 <= bayes_error:
        print("Accept hypothesis: y is a polygonal of x")

    print("Testing error of model 1 %.3f", loss(model1.predict(X_train), y_train))
    print("Testing error of model 2 %.3f", loss(model2.predict(X_train), y_train))
    print("Testing error of model 3 %.3f", loss(model3.predict(X_train), y_train))

