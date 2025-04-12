##ΜΑΡΙΑ ΜΗΤΚΑ
##tem2884
##Lab2 Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import  solve
from sklearn.linear_model import LogisticRegression

# Read training or test data from file
def read_data(filename):
    with open(filename) as fd:
        lines = fd.readlines()

    X = []
    y = []

    for line in lines[1:]:  # Skip header
        data = [float(x) for x in line.split()]
        X.append(data[:-1])
        y.append(data[-1])

    from sklearn.preprocessing import StandardScaler #normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, np.asarray(y)

##compute logistic or sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

##compute loss function
def cost_function(theta, X, y):
    m = y.size
    h = sigmoid(X @ theta)
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m

##comute gradient
def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X @ theta)
    return (1 / m) * X.T @ (y - h)

##compute hessian
def hessian(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    W = np.diag(h * (1 - h))
    return (1 / m) *( X.T @ W @ X)

## synartisi Newton
def newton_method(X, y, epsilon=1e-5, max_iters=100):
    theta = np.zeros(X.shape[1])
    i = 0
    print(f"\n[Parameters Newton] max_iters = {max_iters}, epsilon = {epsilon:.1e}")

    while i < max_iters:
        grad = gradient(theta, X, y)
        H = hessian(theta, X, y)

        print(f"  epanalipsi {i + 1}, Hessian:\n{H}\n")

        theta_new = theta +solve(H, grad)

        if np.linalg.norm(theta_new - theta) < epsilon:
            print(f'|| Θ_(k+1) - Θ_k || < ε ikanopoihtai sto vhma {i}')
            break

        theta = theta_new
        i += 1
        if i == max_iters:
            print('eftase to megisto orio epanalipsewn')
            break

    return theta

datasets = [("set1", "set1_train.txt", "set1_test.txt"),
            ("set2", "set2_train.txt", "set2_test.txt")]
epsilons = [1e-4, 1e-5]

for set_name, train_file, test_file in datasets: #gia na ektelountai oi methodoi gia kathe dataset
    for epsilon in epsilons:
        print(f"\n--- Train for {set_name} with epsilon = {epsilon:.0e} ---")
        X_train, y_train= read_data(train_file)
        X_test, y_test = read_data(test_file)

        ##ekpaideysh me Newton method
        theta = newton_method(X_train, y_train, epsilon=epsilon)

        #ypologizw pithanotites kai provlepseis
        probabilities = sigmoid(X_test @ theta)
        preds = (probabilities >= 0.5).astype(int)

        print(f"\n[Newton] probabilities:\n{probabilities}")


        # Scikit-learn Logistic Regression
        clf = LogisticRegression()
        clf.fit(X_train[:, 1:], (y_train > 0).astype(int))
        sklearn_preds = clf.predict(X_test[:, 1:])
        sklearn_probs = clf.predict_proba(X_test[:, 1:])[:, 1]

        print(f"[scikit-learn] probabilities:\n{sklearn_probs}")

        # Plot
        plt.figure()
        dbdry = -theta[0] * X_test[:, 0] / theta[1]

        xmin = np.min(X_test[:, 0])
        xmax = np.max(X_test[:, 0])
        ymin = np.min(X_test[:, 1])
        ymax = np.max(X_test[:, 1])

        plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=preds, cmap='bwr', alpha=0.9, label='Predictions (Newton)')
        plt.plot(X_test[:, 0], dbdry, 'k--', lw=1., label='Decision boundary (Newton)')
        plt.title(f'{set_name} | epsilon={epsilon:.0e}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()


