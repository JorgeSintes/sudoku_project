import os

import cv2
import torch
import numpy as np

from sudoku_io import show_image

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

def load_char74():
    if os.path.exists(os.path.join("img", "char74.npz")):
        data = np.load(os.path.join("img", "char74.npz"))
    else:
        process_char74()
        data = np.load(os.path.join("img", "char74.npz"))

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

def process_char74():
    sample_dirs = os.listdir(os.path.join("img", "char74_raw", "Fnt"))
    sample_dirs.sort()

    X = np.zeros((10160, 1, 28, 28), dtype=np.float32)
    y = np.zeros((10160,), dtype=np.uint8)

    for i, sample_dir in enumerate(sample_dirs[:10]):
        images = os.listdir(os.path.join('img', 'char74_raw', 'Fnt', sample_dir))
        for j, image in enumerate(images):
            img = cv2.imread(os.path.join('img', 'char74_raw', 'Fnt', sample_dir, image), 0)
            img = cv2.resize(img, (28,28))
            X[i*1016+j, 0, :, :] = img/255
            y[i*1016+j] = i
    
    X_train = []
    X_valid = []
    X_test = []
    y_train = []
    y_valid = []
    y_test = []
    
    for i in range(10):
        images = X[i*1016:(i+1)*1016,:,:,:]
        labels = y[i*1016:(i+1)*1016]

        X_train.append(images[:725,:,:,:])
        X_valid.append(images[725:870,:,:,:])
        X_test.append(images[870:,:,:,:])
        y_train.append(labels[:725])
        y_valid.append(labels[725:870])
        y_test.append(labels[870:])
    
    X_train = np.concatenate(X_train, axis=0)
    X_valid = np.concatenate(X_valid, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    save_dict = {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }

    np.savez(os.path.join("img", "char74.npz"), **save_dict)

if __name__ == "__main__":
    process_char74()
    load_char74()