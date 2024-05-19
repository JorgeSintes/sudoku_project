import enum
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset


class Char74Type(enum.Enum):
    FNT = "Fnt"
    HND = "Hnd"
    GOOD = "GoodImg"
    BAD = "BadImg"

    def __str__(self):
        return self.value

    def get_path(self, base_path: Path) -> Path:
        if self == Char74Type.FNT:
            return base_path / "char74" / "English" / "Fnt"
        elif self == Char74Type.HND:
            return base_path / "char74" / "English" / "Hnd" / "Img"
        elif self == Char74Type.GOOD:
            return base_path / "char74" / "English" / "Img" / "GoodImg" / "Bmp"
        elif self == Char74Type.BAD:
            return base_path / "char74" / "English" / "Img" / "BadImag" / "Bmp"

        raise NotImplementedError(f"Unknown type {self}")


@dataclass
class DataConfig:
    dataset_base_path: Path = Path(__file__).parent / "dataset"
    img_size: int = 28
    use_mnist: bool = True
    use_char74: bool = True
    char74_types: Sequence[Char74Type] = (Char74Type.FNT, Char74Type.HND, Char74Type.GOOD, Char74Type.BAD)


def load_mnist(cfg: DataConfig, train: bool = True) -> Dataset:
    cfg.dataset_base_path.mkdir(exist_ok=True)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: -x),
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


def process_char74(cfg: DataConfig) -> Dataset:
    paths = [ds_type.get_path(cfg.dataset_base_path) for ds_type in cfg.char74_types]

    for path in paths:
        if not path.exists():
            get_char74(cfg)

    total_images = 0
    for i in range(1, 11):
        for path in paths:
            total_images += len(list((path / f"Sample{i:03}").glob("*.png")))

    X = torch.zeros(total_images, 1, cfg.img_size, cfg.img_size, dtype=torch.float32)
    y = torch.zeros(total_images, dtype=torch.uint8)

    idx = 0
    for i in range(10):
        for path in paths:
            for img_path in sorted((path / f"Sample{i+1:03}").iterdir()):
                img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (cfg.img_size, cfg.img_size))
                img = torch.from_numpy(img).type(torch.float32)
                X[idx, 0, :, :] = img / 255 * 2 - 1
                y[idx] = i
                idx += 1

    # torch.save(TensorDataset(X, y), cfg.dataset_base_path / f"char74_{cfg.img_size}.pt")
    return TensorDataset(X, y)


def load_char74(cfg: DataConfig, train: bool = True) -> Dataset:
    if (cfg.dataset_base_path / f"char74_{cfg.img_size}.pt").exists():
        return torch.load(cfg.dataset_base_path / f"char74_{cfg.img_size}.pt")

    return process_char74(cfg)


def load_data(cfg: DataConfig, train: bool = True) -> Dataset:
    datasets = []
    if cfg.use_mnist:
        datasets.append(load_mnist(cfg, train))
    if cfg.use_char74:
        datasets.append(load_char74(cfg, train))
    return ConcatDataset(datasets)


def viz_data(dl: DataLoader):
    X, y = next(iter(dl))
    batch_size = X.shape[0]
    fig, axs = plt.subplots(batch_size, 1, figsize=(12, 6))
    for i, (img, label) in enumerate(zip(X, y)):
        axs[i].set_title(f"Label: {label.item()}")
        axs[i].imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()
