import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class TrainTinyImageNet(Dataset):
    def __init__(self, root, class_to_index, transform=None):
        self.filenames = sorted(
            glob.glob(os.path.join(root, "train", "*", "images", "*.JPEG"))
        )
        self.transform = transform
        self.class_to_index = class_to_index

        if not self.filenames:
            raise FileNotFoundError(
                f"No Tiny-ImageNet training images were found under: {root}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        class_id = image_path.split(os.sep)[-3]
        label = self.class_to_index[class_id]
        if self.transform:
            image = self.transform(image)
        return image, label


class ValTinyImageNet(Dataset):
    def __init__(self, root, class_to_index, transform=None):
        self.filenames = sorted(
            glob.glob(os.path.join(root, "val", "images", "*.JPEG"))
        )
        self.transform = transform
        self.class_by_filename = {}

        annotations = os.path.join(root, "val", "val_annotations.txt")
        with open(annotations, "r", encoding="utf-8") as handle:
            for line in handle:
                filename, class_id = line.rstrip().split("\t")[:2]
                self.class_by_filename[filename] = class_to_index[class_id]

        if not self.filenames:
            raise FileNotFoundError(
                f"No Tiny-ImageNet validation images were found under: {root}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        label = self.class_by_filename[os.path.basename(image_path)]
        if self.transform:
            image = self.transform(image)
        return image, label
