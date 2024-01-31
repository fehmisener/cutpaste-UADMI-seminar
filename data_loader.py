import os
import torch
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from cutpaste import CutPaste


from joblib import Parallel, delayed
from pathlib import Path


class BrainMRI(Dataset):
    """
    Dataset class for loading brain MRI images.

    Parameters:
        - split_dir (str): The directory containing split information and CSV files.
        - pathology (str, optional): The pathology of the images (default is None).
        - size (tuple, optional): The size to which images should be resized (default is (256, 256)).
        - mode (str, optional): The mode of the dataset ('train' or 'test', default is 'train').
        - cutpaste_type (str, optional): The type of cut-paste augmentation (default is 'binary').
        - data_display_mode (bool, optional): Display mode for cut-paste data augmentation (default is False).
            If True, a rectangular border is added to the augmented image.
        - localization (bool, optional): Localization mode (default is False).

    Methods:
        -  __init__(): Initializes the BrainMRI dataset.
        - __len__(): Returns the total number of images in the dataset.
        - __getitem__(idx): Returns an image and its corresponding label based on the index.

    References:
        - FastMRI: An Open Dataset and Benchmarks for Accelerated MRI https://fastmri.med.nyu.edu/
        - IXI Dataset https://brain-development.org/ixi-dataset/
    """

    def __init__(
        self,
        split_dir: str,
        pathology=None,
        size=(256, 256),
        mode="train",
        cutpaste_type="binary",
        data_display_mode=False,
        localization=False,
    ):
        self.mode = mode
        self.size = size
        self.split_dir = split_dir
        self.pathology = pathology
        self.localization = localization
        self.cutpaste = CutPaste(
            type=cutpaste_type, data_display_mode=data_display_mode
        )
        self.transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor()]
        )

        if self.mode == "train" or self.localization:
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
    """
    Dataset class for loading MVTec Anomaly Detection dataset images.

    Parameters:
        - root_dir (str): The root directory of the dataset.
        - defect_name (str): The name of the defect category.
        - size (tuple, optional): The size to which images should be resized (default is (256, 256)).
        - mode (str, optional): The mode of the dataset ('train' or 'test', default is 'train').
        - cutpaste_type (str, optional): The type of cut-paste augmentation (default is 'binary').
        - data_display_mode (bool, optional): Display mode for cut-paste data augmentation (default is False).

    Methods:
        - __init__(): Initializes the MVTecAT dataset.
        - __len__(): Returns the total number of images in the dataset.
        - __getitem__(idx): Returns an image and its label based on the index.

    References:
        - MVTec Anomaly Detection Dataset https://www.mvtec.com/company/research/datasets/mvtec-ad/

    """

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
    """
    Collate function for custom batch processing.

    Parameters:
        - batch (list): A list of batch items.

    Returns:
        - list: A list of torch tensors containing the collated batch.

    """
    img_types = list(zip(*batch))
    return [torch.stack(imgs) for imgs in img_types]
