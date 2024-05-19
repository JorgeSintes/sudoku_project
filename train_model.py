import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import mlflow
import numpy as np
from mlflow.artifacts import tempfile
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from data import load_data, DataConfig, Char74Type
from model import Baseline, DCNN


@dataclass
class TrainConfig:
    data_config: DataConfig
    mlflow_experiment_name: str
    mlflow_uri: str
    batch_size: int
    img_size: int
    model: str
    no_epochs: int
    lr: float
    loss_fn: str
    optimizer: str
    seed: int | None
    valid_every_steps: int
    save_checkpoint: bool


def train(log: logging.Logger, config: TrainConfig):
    if config.seed:
        torch.manual_seed(config.seed)
    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Using {device}")

    train_ds = load_data(config.data_config, train=True)
    test_ds = load_data(config.data_config, train=False)
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True)

    if config.model == "baseline":
        model = Baseline(Baseline.Config(img_size=config.img_size))
    elif config.model == "dcnn":
        model = DCNN(DCNN.Config(img_size=config.img_size))
    else:
        raise NotImplementedError("SudokuNet is the only supported model")
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else:
        raise NotImplementedError("Adam is the only supported optimizer")
    if config.loss_fn == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Cross-entropy is the only supported loss_fn")

    model.to(device)
    loss_fn.to(device)

    # with mlflow.start_run():
    mlflow.log_params(asdict(config))

    step = 0
    train_acc = []
    test_acc = []
    best_test_acc = 0
    train_acc_batches = []
    for epoch in range(config.no_epochs):
        log.info(f"Epoch: {epoch}")
        model.train()
        for train_X, train_y in train_dl:
            output = model(train_X.to(device)).to("cpu")
            loss = loss_fn(output, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Step {step}: Loss: {loss.item()}")
            step += 1
            predictions = output.max(1)[1]
            train_acc_batches.append(accuracy_score(train_y, predictions))

            if step % config.valid_every_steps == 0:
                train_acc.append(np.mean(train_acc_batches))
                train_acc_batches = []
                test_acc_batches = []
                with torch.no_grad():
                    model.eval()
                    for test_X, test_y in test_dl:
                        predictions = model.inference(test_X.to(device)).to("cpu")
                        test_acc_batches.append(accuracy_score(test_y, predictions))

                test_acc.append(np.mean(test_acc_batches))

                log.info(f"Step {step}:")
                log.info(f"\tTraining acc: {train_acc[-1]}")
                log.info(f"\tTest acc: {test_acc[-1]}")
                mlflow.log_metrics(
                    {
                        "train_acc": train_acc[-1],
                        "test_acc": test_acc[-1],
                    },
                    step=step,
                )

                if config.save_checkpoint and test_acc[-1] > best_test_acc:
                    best_test_acc = test_acc[-1]
                    checkpoint = {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "accuracy": test_acc,
                    }
                    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
                        torch.save(checkpoint, tmp.name)
                        mlflow.log_artifact(tmp.name, artifact_path="model_checkpoints")

        log.info("Finished training!")


#     def test_sudoku(self, path=os.path.join("img", "sudoku.jpeg")):
#         sudoku_images = get_sudoku_images(path, img_size=self.img_size)
#         for i in range(9):
#             for j in range(9):
#                 digit_input = torch.Tensor(sudoku_images[i, j, :, :]).reshape(
#                     1, 1, self.img_size, self.img_size
#                 )
#                 prediction = self.model.inference(digit_input.to(self.device)).to("cpu")
#                 self.log.info(f"Row {i}, Column {j}. Pred: {prediction[0]}")
#                 show_image(sudoku_images[i, j, :, :])
#
#     def test_char(self):
#         self.test_loader
#
#     def save_weights(self, path):
#         self.model.save_weights(path)
#
#     def load_weights(self, path):
#         self.model.load_weights(path)
#


def make_logger():
    logger = logging.getLogger("sudoku")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    log = make_logger()
    config = TrainConfig(
        data_config=DataConfig(
            dataset_base_path=Path(__file__).parent / "dataset",
            img_size=28,
            use_mnist=True,
            use_char74=True,
            char74_types=(Char74Type.FNT, Char74Type.HND, Char74Type.GOOD, Char74Type.BAD),
            char74_cached_data=True,
            char74_ds_split=0.8,
            seed=42,
        ),
        mlflow_experiment_name="First trial",
        mlflow_uri="http://127.0.0.1:5000",
        batch_size=32,
        img_size=28,
        model="baseline",
        no_epochs=10,
        lr=1e-3,
        loss_fn="cross_entropy",
        optimizer="adam",
        valid_every_steps=100,
        seed=42,
        save_checkpoint=True,
    )
    train(log, config)
