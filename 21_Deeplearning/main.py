import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


# Normalize input data by using mean normalizaton
def mean_normalization(X):
    N = len(X)

    maxi = np.max(X)
    mini = np.min(X)
    avg = np.mean(X)
    X = (X - avg) / (maxi - mini)
    X_b = np.c_[np.ones((N, 1)), X]
    return X_b, maxi, mini, avg

def stochastic_gradient_descent(X_b, y, n_epochs=50, learning_rate=0.01):
    N, d_plus1 = X_b.shape
    # Step 1: Init parameters
    thetas = np.asarray([[1.16270837], [-0.81960489],
                         [1.39501033], [0.29763545]])
    thetas_path = [thetas.copy()]

    losses = []
    # Step 2-6: SGD loop
    for epoch in range(n_epochs):
        for i in range(N):
            random_index = i
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # Step 3: prediction
            y_hat = np.sum(xi @ thetas)

            # Step 4: Compute losses
            li = (y_hat - yi) ** 2

            # Step 5: Compute gradient
            dl_dwi = 2 * xi[:, 1:] * (y_hat - yi)
            dl_db = 2 * (y_hat - yi)
            dl_dtheta = np.hstack([dl_db, dl_dwi]).T

            # Step 6: Update parameters
            thetas = thetas - learning_rate * dl_dtheta

            thetas_path.append(thetas.copy())
            losses.append(li[0][0])

    return thetas_path, losses

def mini_batch_gradient_descent(X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01):
    N, d_plus1 = X_b.shape
    # Step 1: Init parameters
    thetas = np.asarray([[1.16270837], [-0.81960489],
                         [1.39501033], [0.29763545]])
    thetas_path = [thetas.copy()]
    losses = []

    for epoch in range(n_epochs):
        # shuffle data
        shuffled_indices = np.random.permutation(N)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, N, minibatch_size):
            xi = X_b_shuffled[i:i + minibatch_size]  # (m, d+1)
            yi = y_shuffled[i:i + minibatch_size]  # (m, 1)

            # Step 3: prediction
            y_hat = xi @ thetas
            # Step 4: MSE loss for minibatch
            loss = 1 / minibatch_size * np.sum((y_hat - yi)**2)
            # Step 5: gradient for minibatch
            # dl_dwi = 2 / minibatch_size * xi[:, 1:].T @ (y_hat - yi)
            # dl_db = 2 / minibatch_size * (np.ones_like(xi[:, 0].T) @ (y_hat - yi))
            dl_dwi = 2 / minibatch_size * np.sum((y_hat - yi) * xi[:, 1:], axis=0)
            dl_db = 2 / minibatch_size * np.sum(y_hat - yi, axis=0)
            dl_dtheta = np.hstack([dl_db, dl_dwi]).reshape(-1, 1)

            dl_dtheta = 2 / minibatch_size * xi.T @ (y_hat - yi)
            # Step 6: update
            thetas = thetas - learning_rate * dl_dtheta
            # log
            thetas_path.append(thetas.copy())
            losses.append(loss)

    return thetas_path, losses


def batch_gradient_descent(X_b, y, n_epochs=100, learning_rate=0.01):
    N, d_plus1 = X_b.shape
    # Step 1: Init parameters
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033],
    [0.29763545]])
    thetas_path = [thetas.copy()]
    losses = []
    xi = X_b
    yi = y
    for epoch in range(n_epochs):
        # Step 3: prediction
        y_hat = xi @ thetas
        # Step 4: MSE loss for minibatch
        loss = 1 / N * np.sum((y_hat - yi) ** 2)
        # Step 5: gradient for minibatch
        # dl_dwi = 2 / N * xi[:, 1:].T @ (y_hat - yi)
        # dl_db = 2 / N * (np.ones_like(xi[:, 0].T) @ (y_hat - yi))
        dl_dwi = 2 / N * np.sum((y_hat - yi) * xi[:, 1:], axis=0)
        dl_db = 2 / N * np.sum(y_hat - yi, axis=0)
        dl_dtheta = np.hstack([dl_db, dl_dwi]).reshape(-1, 1)

        # dl_dtheta = 2 / N * xi.T @ (y_hat - yi)
        # Step 6: update
        thetas = thetas - learning_rate * dl_dtheta
        # log
        thetas_path.append(thetas.copy())
        losses.append(loss)

    return thetas, losses

if __name__ == '__main__':
    data_dir = r'C:\Users\User\Desktop\Akselos_Project\my\AI_VN\21_Deeplearning\Advertising.csv'
    # dataset
    data = genfromtxt(data_dir, delimiter=",", skip_header=1)
    N = data.shape[0]
    X = data[:, :3]
    y = data[:, 3:]
    X_b, maxi, mini, avg = mean_normalization(X)
    # sgd_theta, losses = stochastic_gradient_descent(X_b, y, n_epochs=50, learning_rate=0.01)
    # print(losses)
    # # In loss cho 500 bước đầu
    # x_axis = list(range(500))
    # plt.plot(x_axis, losses[:500], color="r")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("SGD Loss per Step (first 500 updates)")
    # plt.show()

    # mbgd_thetas, losses = mini_batch_gradient_descent(X_b, y, n_epochs=
    # 50, minibatch_size=20,
    #                                                   learning_rate=0.01)
    # # visualize
    # x_axis = list(range(200))
    # plt.plot(x_axis, losses[:200], color="r")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("Mini-batch Gradient Descent Loss (first 200 updates)")
    # plt.show()

    bgd_thetas, losses = batch_gradient_descent(X_b, y, n_epochs=100,
                                                learning_rate=0.01)
    # visualize
    x_axis = list(range(100))
    plt.plot(x_axis, losses[:100], color="r")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Batch Gradient Descent Loss")
    plt.show()