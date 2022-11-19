import os
import logging

import torch
import numpy as np
from termcolor import colored
from sklearn.metrics import accuracy_score

from data import load_mnist
from model import SudokuNet
from sudoku_io import show_image, get_sudoku_images

class Trainer():
    def __init__(self, log: logging.Logger, batch_size: int = 64):
        self.log = log
        self.batch_size = batch_size

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.log.info(f"Using {self.device}!")

        self.create_data_loaders(dataset="mnist")

        self.model = SudokuNet()
        self.model.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn.to(self.device)

    def create_data_loaders(self, dataset="mnist"):
        if dataset == "mnist":
            data_dict = load_mnist()

        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_valid = data_dict["X_valid"]
        y_valid = data_dict["y_valid"]
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"]

        train_set = torch.utils.data.TensorDataset(X_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid_set = torch.utils.data.TensorDataset(X_valid, y_valid)
        self.valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_set = torch.utils.data.TensorDataset(X_test, y_test)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def train(self, no_epochs: int = 5, lr: float = 1e-3, valid_every_steps:int = 500):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        step = 0
        train_acc = []
        valid_acc = []
        for epoch in range(no_epochs):
            train_acc_batches = []
            self.model.train()
            for inputs, targets in self.train_loader:
                output = self.model(inputs.to(self.device)).to("cpu")
                loss = self.loss_fn(output, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                step += 1
                predictions = output.max(1)[1]
                train_acc_batches.append(accuracy_score(targets, predictions))

                if step % valid_every_steps == 0:
                    train_acc.append(np.mean(train_acc_batches))
                    train_acc_batches = []

                    valid_acc_batches = []
                    with torch.no_grad():
                        self.model.eval()
                        for inputs, targets in self.valid_loader:
                            predictions = self.model.inference(inputs.to(self.device)).to("cpu")
                            valid_acc_batches.append(accuracy_score(targets, predictions) * inputs.shape[0])
                    
                    valid_acc.append(np.sum(valid_acc_batches)/len(self.valid_loader.dataset))
                    self.log.info(f"Step {step}:")
                    self.log.info(f"\tTraining acc: {train_acc[-1]}")
                    self.log.info(f"\tValidation acc: {valid_acc[-1]}")
            
        self.log.info("Finished training!")

    def test(self):
        test_acc_batches = []
        with torch.no_grad():
            self.model.eval()
            for inputs, targets in self.test_loader:
                predictions = self.model.inference(inputs.to(self.device)).to("cpu")
                test_acc_batches.append(accuracy_score(targets, predictions) * inputs.shape[0])
        
        test_acc = np.sum(test_acc_batches) / len(self.test_loader.dataset)
        self.log.info(f"Test acc: {test_acc}")
    
    def test_sudoku(self, path=os.path.join("img", "sudoku.jpeg")):
        sudoku_images = get_sudoku_images(path)
        for i in range(9):
            for j in range(9):
                digit_input = torch.Tensor(sudoku_images[i, j, :, :]/255).reshape(1,1,28,28)
                prediction = self.model.inference(digit_input.to(self.device)).to("cpu")
                self.log.info(f"Row {i}, Column {j}. Pred: {prediction[0]}")
                show_image(sudoku_images[i,j,:,:])
    
    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)
        


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    from : https://stackoverflow.com/questions/1343227/can-pythons-logging-format-be-modified-depending-on-the-message-log-level
    """
    format_strings = {
        logging.DEBUG: colored("[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s", color='cyan',attrs=['bold']),
        logging.INFO: colored("[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s", color='white'),
        logging.WARNING: colored("[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s", color='yellow'),
        logging.ERROR: colored("[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s", color='red'),
        logging.CRITICAL: colored("[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s", color='white',on_color='on_red')
    }
    def format(self, record):
        log_fmt = self.format_strings.get(record.levelno)
        formatter = logging.Formatter(log_fmt,datefmt='%H:%M:%S')
        return formatter.format(record)
                  
def make_logger():
    logger = logging.getLogger("sudoku")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

def example():
    log = make_logger()
    trainer = Trainer(log=log, batch_size=64)
    trainer.load_weights("weights/sudokunet.pt")
    trainer.test()
    trainer.test_sudoku()

if __name__ == "__main__":
    example()