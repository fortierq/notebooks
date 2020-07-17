import numpy as np
import matplotlib.pyplot as plt

def points(n): 
    X1 = np.random.multivariate_normal([4, 3], 5*np.eye(2), n)
    X2 = np.random.multivariate_normal([-2, -1], 5*np.eye(2), n)
    X = np.concatenate((X1, X2))
    Y = np.array([0]*n + [1]*n)
    p = np.random.permutation(2*n)
    return X[p].T, Y[p].T
    
def plot_points(X, Y):
    plt.scatter(X[0], X[1], c = Y, s = 100,cmap=plt.cm.Spectral)
    
def init(n):
    X, Y = points(n)
    w = np.array([[0],[1]])
    b = 0
    return w, b, X, Y #{'X': X, 'Y': Y, 'w': w, 'b': b}
    
def sigmoid(w, b, X):
    return 1/(1 + np.exp(-(np.dot(w.T, X) + b)))

def predict(w, b, X):
    return np.round(sigmoid(w, b, X))

def accuracy(w, b, X, Y):
    P = predict(w, b, X)
    return float(np.sum(P == Y, axis = 1)/len(Y))
    
def plot_decision_boundary(pred_func, X, Y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c = Y, s = 100, cmap=plt.cm.Spectral)
    
def optimize(w, b, X, Y, a, n):
    m = X.shape[1] # nombre de points
    plot_points(X, Y)
    plt.draw()
    plt.pause(2)
    for i in range(n):
        A = sigmoid(w, b, X)
        if i % 50 == 0:
            plt.clf()
            plot_decision_boundary(lambda x: predict(w, b, x.T), X.T, Y)
            print("Accuracy: ", accuracy(w, b, X, Y))
            plt.pause(2) 
        dw = np.dot(X, (A - Y).T)/m
        db = np.sum(A - Y)/m
        w = w - a*dw
        b = b - a*db
    return w, b 

w, b, X, Y = init(15)
w, b = optimize(w, b, X, Y, 0.01, 400)

