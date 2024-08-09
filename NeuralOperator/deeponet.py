import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import deepxde as dde
from deepxde.backend import tf


def get_data(filename, n_train, n_test):
    data = scipy.io.loadmat(filename)
    x = data["x_coor"].astype(np.float32)
    y = data["y_coor"].astype(np.float32)
    t = data["t"].astype(np.float32)
    Vx = data["Vx"].astype(np.float32)
    Vy = data["Vy"].astype(np.float32)
    
    u = np.stack((Vx, Vy), axis=-1)  # Shape: (N, k, n_points, 2)
    u_final = u[:, 2, :, :]  # Shape: (N, n_points, 2)

    # u_flattened = u.reshape(u.shape[0], u.shape[1], -1) #Input branch only recieves the 1D array
    # Branch: initial velocity field
    branch = u[:, 0:2, :, :]  # Shape: (N, k = 0,1, n_points, 2)
    branch = branch.reshape(branch.shape[0], -1)  # Shape: (N, 2 * n_points)
    # Trunk: spatial coordinates grid
    xy = np.vstack((np.ravel(x), np.ravel(y))).T  # N x 2 (x, y)：grid points
    
    # Output: velocity field at the final time step


    # Split into training and testing data
    branch_train = branch[:n_train]
    xy_train = xy
    u_final_train = u_final[:n_train]

    branch_test = branch[-n_test:]
    xy_test = xy
    u_final_test = u_final[-n_test:]

    return (branch_train, xy_train), u_final_train, (branch_test, xy_train), u_final_test

def main():
    n_train = 100  # Adjust based on your dataset size
    n_test = 60  # Adjust based on your dataset size
    nx = 6400

    data_path = "/home/liruixiang/data/gepup/train_data_new.mat"
    x_train, u_final_train, x_test, u_final_test = get_data(data_path, n_train, n_test)
    
    # Split u_final_train and u_final_test into Vx and Vy parts
    u_final_train_vx = u_final_train[:, :, 0]
    u_final_train_vy = u_final_train[:, :, 1]
    u_final_test_vx = u_final_test[:, :, 0]
    u_final_test_vy = u_final_test[:, :, 1]
    
    # train_x = [branch_train, xy_train]
    # test_x = [branch_test, xy_test]

    # Create datasets for Vx and Vy
    data_vx = dde.data.Triple(
        x_train, u_final_train_vx,
        x_test, u_final_test_vx
    )
    data_vy = dde.data.Triple(
        x_train, u_final_train_vy,
        x_test, u_final_test_vy
    )

    # 创建两个子网络，分别生成Vx和Vy
    net_vx = dde.maps.DeepONetCartesianProd(
        [nx * 4, 264, 264], [2, 264, 264], "relu", "Glorot normal"
    )

    net_vy = dde.maps.DeepONetCartesianProd(
        [nx * 4, 264, 264], [2, 264, 264], "relu", "Glorot normal"
    )

    # 创建模型
    model_vx = dde.Model(data_vx, net_vx)
    model_vy = dde.Model(data_vy, net_vy)

    # 编译模型
    model_vx.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    model_vy.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )

    # 训练模型
    losshistory_vx, train_state_vx = model_vx.train(epochs=50000, batch_size=None)
    losshistory_vy, train_state_vy = model_vy.train(epochs=50000, batch_size=None)

    # 预测
    y_pred_vx = model_vx.predict(data_vx.test_x)
    y_pred_vy = model_vy.predict(data_vy.test_x)

    # 保存预测结果
    np.savetxt("y_pred_vx_deeponet.dat", y_pred_vx[0].reshape(nx, 1))
    np.savetxt("y_pred_vy_deeponet.dat", y_pred_vy[0].reshape(nx, 1))
    np.savetxt("y_true_vx_deeponet.dat", data_vx.test_y[0].reshape(nx, 1))
    np.savetxt("y_true_vy_deeponet.dat", data_vy.test_y[0].reshape(nx, 1))
    np.savetxt("y_error_vx_deeponet.dat", (y_pred_vx[0] - data_vx.test_y[0]).reshape(nx, 1))
    np.savetxt("y_error_vy_deeponet.dat", (y_pred_vy[0] - data_vy.test_y[0]).reshape(nx, 1))

if __name__ == "__main__":
    # Enable memory growth for GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(e)
    
    main()
