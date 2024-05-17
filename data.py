import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset


@dataclass
class DataConfig:
    dataset_base_path: Path = Path(__file__).parent / "dataset"
    img_size: int = 28
    use_mnist: bool = True
    use_char74: bool = True


def load_mnist(cfg: DataConfig, train: bool = True) -> Dataset:
    cfg.dataset_base_path.mkdir(exist_ok=True)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    ds = torchvision.datasets.MNIST(
        cfg.dataset_base_path.as_posix(), train=train, download=True, transform=transform
    )
    return ds


def get_char74(cfg: DataConfig):
    dataset_path = cfg.dataset_base_path / "char74"
    dataset_path.mkdir(exist_ok=True)
    for file in ["EnglishImg.tgz", "EnglishHnd.tgz", "EnglishFnt.tgz"]:
        if not (dataset_path / "English" / file[7:10]).exists():
            os.system(
                f"wget -P {dataset_path.as_posix()} https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/{file}"
            )
            os.system(f"tar -xvzf {dataset_path.as_posix()}/{file} -C {dataset_path.as_posix()}")
            os.system(f"rm {dataset_path.as_posix()}/{file}")


def process_char74(cfg: DataConfig):
    fnt_path = cfg.dataset_base_path / "char74" / "English" / "Fnt"
    if not fnt_path.exists():
        get_char74(cfg)

    total_images = 0
    for i in range(1, 11):
        total_images += len(list((fnt_path / f"Sample{i:03}").glob("*.png")))

    X = torch.zeros(total_images, 1, cfg.img_size, cfg.img_size, dtype=torch.float32)
    y = torch.zeros(total_images, dtype=torch.uint8)

    for i in range(10):
        for j, img_path in enumerate(sorted((fnt_path / f"Sample{i+1:03}").iterdir())):
            img = cv2.imread(img_path.as_posix(), 0)
            img = cv2.resize(img, (cfg.img_size, cfg.img_size))
            img = torch.from_numpy(img).type(torch.float32)
            X[i * 1016 + j, 0, :, :] = img / 255 * 2 - 1
            y[i * 1016 + j] = i

    torch.save(TensorDataset(X, y), cfg.dataset_base_path / f"char74_{cfg.img_size}.pt")


def load_char74(cfg: DataConfig, train: bool = True) -> Dataset:
    if not (cfg.dataset_base_path / f"char74_{cfg.img_size}.pt").exists():
        process_char74(cfg)

    return torch.load(cfg.dataset_base_path / f"char74_{cfg.img_size}.pt")


def load_data(cfg: DataConfig, train: bool = True) -> Sequence[Dataset]:
    datasets = []
    if cfg.use_mnist:
        datasets.append(load_mnist(cfg, train))
    if cfg.use_char74:
        datasets.append(load_char74(cfg, train))
    return datasets


def create_data_loader(datasets: Sequence[Dataset], batch_size: int, shuffle: bool) -> DataLoader:
    sets = ConcatDataset(datasets)
    return DataLoader(sets, batch_size=batch_size, shuffle=shuffle)


def viz_data(dl: DataLoader, batch_size: int = 5):
    fig, axs = plt.subplots(batch_size, 1, figsize=(12, 6))
    X, y = next(iter(dl))
    for i, (img, label) in enumerate(zip(X, y)):
        axs[i].title(label)
        axs[i].imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()
