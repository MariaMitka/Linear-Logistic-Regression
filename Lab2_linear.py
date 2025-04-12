##ΜΑΡΙΑ ΜΗΤΚΑ
##tem2884
##Lab2 Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

#synarthsh gia na kanonikopoihsei ta features kai ton stoxo
#vriskontas ta mean kai standard deviations
def normalize(X, Y):
    mean_X = np.mean(X, axis=0)
    sigma_X = np.std(X, axis=0)
    X = (X - mean_X) / sigma_X


    mean_Y = np.mean(Y)
    sigma_Y = np.std(Y)
    Y = (Y - mean_Y) / sigma_Y

    return X, Y, mean_X, sigma_X, mean_Y, sigma_Y

##synarthsh gia na anevasw ta data
def dataset(filename, mean_X=None, sigma_X=None, mean_Y=None, sigma_Y=None):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]

    if mean_X is None and sigma_X is None and mean_Y is None and sigma_Y is None:
        X, Y, mean_X, sigma_X, mean_Y, sigma_Y = normalize(X, Y) #gia ta dedomena ekpaideyshs
    else: ##gia ta dedomena dokimhs
        X = (X -mean_X) / sigma_X
        Y = (Y -mean_Y)/ sigma_Y

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    return X, Y, mean_X, sigma_X, mean_Y, sigma_Y

#@synarthsh ypologismoy kostoys
def cost_func(theta, X, Y):
    N =Y.size
    h_theta = X @ theta
    error = h_theta - Y
    return np.sum(error ** 2) * (1/(2*N))

##synarthsh gia to gradient
def gradient_func(theta, X, Y):
    N=Y.size
    return (X.T @ (X @ theta - Y)) / N

##synarthsh gia th methodo gradient descent
def gradient_descent(theta,X, Y, a, max_iters, eps, delta):
    cost = cost_func(theta,X, Y)
    l = []
    i = 0

    while i < max_iters:
        theta_new = theta - a * gradient_func(theta, X, Y)
        cost_new = cost_func(theta_new, X, Y)
        l.append((theta_new , cost_new))
        if abs(cost_new) < delta:
            print(f"stamatise ston deikth {i} logw |J(θ)|<δ")
            break
        if la.norm(theta_new - theta) < eps:
            print(f"stamatise ston deikth {i} logw ||θ_{i+1} - θ_{i}||")
            break

        theta = theta_new
        i +=1

        if i == max_iters:
            print("H epanalipsi eftase sto megisto deikth")

    return l, i

##main programma ylopoihshs
if __name__ == '__main__':

    X_train, Y_train, mean_X, sigma_X, mean_Y, sigma_Y = dataset('car_train.txt')


    a = 0.01
    max_iters = 10000
    eps  = 1e-6
    delta = 1e-5
    print(f"\nParametroi gia gradient")
    print(f"learning rate = {a}")
    print(f"max iters = {max_iters}")
    print(f"epsilon= {eps:.5e}")
    print(f"delta = {delta:.5e}")

##initialize theta
    theta = np.zeros(5)
    ##ekpaideysh modeloy
    l, iters = gradient_descent(theta, X_train, Y_train, a, max_iters, eps, delta)

    print()
    print(f"arithmos epanalipsewn = {iters}")

    theta = l[-1][0] #teliki timh theta k kostoys
    print(f'theta = ', end='')
    for c in theta: print(f'{c:.6f} ', end='')
    print()
    print(f'J(θ) = {l[-1][1]:.6f}\n')

    cost = [h[1] for h in l]  ##grafima toy kostoys se kathe epanalipsi
    plt.plot(cost, 'k-', label=r'$J(\theta)$')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost function')
    plt.title('Cost development')
    plt.legend()
    plt.show()

    ##dokimh modeloy me ta dedomena dokimhs
    X_test, Y_test, _, _, _, _ = dataset('car_test.txt', mean_X, sigma_X, mean_Y, sigma_Y)
    error_theta = la.norm(X_test @ theta - Y_test)
    print(f"\n test: E_θ = {error_theta:.6f}")

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression().fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)

    err_theta = la.norm(y_pred - Y_test)
    print(f'Test (sklearn): E_θ = {err_theta:.6f}')