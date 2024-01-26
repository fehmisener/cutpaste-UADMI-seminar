import os
import torch
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from cutpaste import CutPaste


from joblib import Parallel, delayed
from pathlib import Path


class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


class BrainMRI(Dataset):
    def __init__(
        self,
        split_dir: str,
        pathology=None,
        size=(256, 256),
        mode="train",
        cutpaste_type="binary",
        data_display_mode=False,
    ):
        self.mode = mode
        self.size = size
        self.split_dir = split_dir
        self.pathology = pathology
        self.cutpaste = CutPaste(
            type=cutpaste_type, data_display_mode=data_display_mode
        )
        self.transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor()]
        )

        if self.mode == "train":
            # Data train mode
            train_files_ixi = pd.read_csv(
                os.path.join(split_dir, "ixi_normal_train.csv")
            )["filename"].tolist()
            train_files_fastMRI = pd.read_csv(
                os.path.join(split_dir, "normal_train.csv")
            )["filename"].tolist()

            self.image_names = train_files_ixi + train_files_fastMRI
            print(
                f"Using {len(train_files_ixi)} IXI images "
                f"and {len(train_files_fastMRI)} fastMRI images. "
                f"Total {len(self.image_names)} images for training."
            )
        else:
            # Data test mode
            self.image_names = pd.read_csv(
                os.path.join(split_dir, f"{self.pathology}.csv")
            )["filename"].tolist()
            normal_test_paths = pd.read_csv(os.path.join(split_dir, "normal_test.csv"))[
                "filename"
            ].tolist()

            self.normal_start_idx = len(self.image_names)
            self.image_names.extend(normal_test_paths)
            print(
                f"Using {self.normal_start_idx} abnormal images "
                f"and {len(self.image_names) - self.normal_start_idx} normal images. "
                f"Total {len(self.image_names)} images for testing."
            )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(self.image_names[idx]).convert("RGB")

        if self.mode == "train":
            cutpaste_augmentation = self.cutpaste(img)
            transformed = (self.transform(image) for image in cutpaste_augmentation)
            return transformed
        else:
            img = self.transform(img)
            return img, idx <= self.normal_start_idx


class MVTecAT(Dataset):

    def __init__(
        self,
        root_dir,
        defect_name,
        size=(256, 256),
        mode="train",
        cutpaste_type="binary",
        data_display_mode=False,
    ):
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.mode = mode
        self.size = size

        self.cutpaste = CutPaste(
            type=cutpaste_type, data_display_mode=data_display_mode
        )
        self.transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor()]
        )

        # find test images
        if self.mode == "train":
            self.image_names = list(
                (self.root_dir / defect_name / "train" / "good").glob("*.png")
            )
            print(f"Total {len(self.image_names)} images.")
        else:
            # test mode
            self.image_names = list(
                (self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png"))
            )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(self.image_names[idx]).convert("RGB")

        if self.mode == "train":
            cutpaste_augmentation = self.cutpaste(img)
            transformed = (self.transform(image) for image in cutpaste_augmentation)
            return transformed
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = self.transform(img)
            return img, label != "good"


def collate_function(batch):
    img_types = list(zip(*batch))
    return [torch.stack(imgs) for imgs in img_types]
