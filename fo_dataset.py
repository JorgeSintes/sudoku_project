import random
from dataclasses import dataclass
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.data as foud
from data import Char74Type, DataConfig


@dataclass
class ImageInfo:
    path: Path
    ground_truth: str
    dataset_type: Char74Type


class Char74DatasetImporter(foud.LabeledImageDatasetImporter):
    def __init__(
        self,
        cfg: DataConfig,
        shuffle=False,
        seed=None,
        max_samples=None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=cfg.dataset_base_path,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )
        self.cfg = cfg
        self.no_samples = 0
        self.max_samples = max_samples
        self.images: list[ImageInfo] = []

        for char74_type in cfg.char74_types:
            path = char74_type.get_path(cfg.dataset_base_path)
            for i in range(10):
                for img_path in sorted((path / f"Sample{i+1:03}").iterdir()):
                    self.images.append(ImageInfo(img_path, str(i), char74_type))

        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.images)

        self._iter_images = iter(self.images)

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.images), self.max_samples)
        return len(self.images)

    def __next__(self):
        """Returns information about the next sample in the dataset.

        Returns:
            an  ``(image_path, image_metadata, label)`` tuple, where

            -   ``image_path``: the path to the image on disk
            -   ``image_metadata``: an
                :class:`fiftyone.core.metadata.ImageMetadata` instances for the
                image, or ``None`` if :meth:`has_image_metadata` is ``False``
            -   ``label``: an instance of :meth:`label_cls`, or a dictionary
                mapping field names to :class:`fiftyone.core.labels.Label`
                instances, or ``None`` if the sample is unlabeled

        Raises:
            StopIteration: if there are no more samples to import
        """
        if self.no_samples == self.max_samples:
            raise StopIteration

        image_info = next(self._iter_images)
        self.no_samples += 1

        video_metadata = fo.ImageMetadata.build_for(image_info.path.as_posix())
        labels = {
            "filepath": image_info.path.as_posix(),
            "label": image_info.ground_truth,
            "dataset_type": image_info.dataset_type.value,
        }

        return image_info.path.as_posix(), video_metadata, labels

    @property
    def has_dataset_info(self):
        """Whether this importer produces a dataset info dictionary."""
        # Return True or False here
        return False

    @property
    def has_image_metadata(self):
        """Whether this importer produces
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each image.
        """
        # Return True or False here
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this
        importer.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            importer is guaranteed to return labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the importer can produce a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the importer will return label dictionaries with keys
            and value-types specified by this dictionary. Not all keys need be
            present in the imported labels
        -   ``None``. In this case, the importer makes no guarantees about the
            labels that it may return
        """
        # Return the appropriate value here
        return {
            "filepath": fo.StringField,
            "ground_truth": fo.StringField,
            "dataset_type": fo.StringField,
        }


if __name__ == "__main__":
    # Make sure mongod is running somewhere
    fo.config.database_uri = "mongodb://localhost:27017/fiftyone"
    available_datasets = fo.list_datasets()
    if "char74" not in available_datasets:
        char74_importer = Char74DatasetImporter(
            DataConfig(
                dataset_base_path=Path(__file__).parent / "dataset",
                char74_types=(Char74Type.FNT, Char74Type.HND, Char74Type.GOOD, Char74Type.BAD),
            ),
            seed=42,
        )
        char74_ds = fo.Dataset.from_importer(char74_importer, "char74", persistent=True)
    if "mnist" not in available_datasets:
        mnist_ds = foz.load_zoo_dataset(
            "mnist", dataset_dir=Path(__file__).parent / "dataset" / "MNIST", persistent=True
        )
    session = fo.launch_app(address="0.0.0.0", port=5151)
    session.wait()
