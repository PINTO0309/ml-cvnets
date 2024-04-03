import argparse

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)


@DATASET_REGISTRY.register(name="hand_point", type="classification")
class HandPointDataset(BaseImageClassificationDataset):
    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        BaseImageClassificationDataset.__init__(
            self,
            opts=opts,
            *args,
            **kwargs,
        )
