import os
import torch
import numpy as np

def load_mnist():
    if not os.path.exists(os.path.join("img", "mnist.npz")):
        os.system("wget -N https://www.dropbox.com/s/qxywaq7nx19z72p/mnist.npz")
        os.system("mv mnist.npz img/")
    data = np.load(os.path.join("img", "mnist.npz"))

    X_train = torch.tensor(data["X_train"]).reshape(-1,1,28,28)
    y_train = torch.tensor(data["y_train"])
    X_valid = torch.tensor(data["X_valid"]).reshape(-1,1,28,28)
    y_valid = torch.tensor(data["y_valid"])
    X_test = torch.tensor(data["X_test"]).reshape(-1,1,28,28)
    y_test = torch.tensor(data["y_test"])

    return_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
    }

    return return_dict


if __name__ == "__main__":
    load_mnist()
