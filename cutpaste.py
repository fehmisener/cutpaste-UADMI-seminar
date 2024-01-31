import random
import numpy as np
from torchvision import transforms

from PIL import ImageDraw


class CutPaste(object):
    """
    Class for performing cut-paste data augmentation on images.

    Parameters:
        - type (str, optional): The type of cut-paste augmentation ('binary' or '3way', default is 'binary').
        - data_display_mode (bool, optional): Display mode for additional border visualization (default is False).

    Attributes:
        - type (str): The type of cut-paste augmentation.
        - data_display_mode (bool): Display mode for additional border visualization.
        - transform (transforms.ColorJitter): Color jitter transformation for patch augmentation.

    Methods:
        - crop_and_paste_patch(image, patch_w, patch_h, rotation=False): Crops and pastes a patch onto the image.
        - cutpaste(image, area_ratio=(0.02, 0.15), aspect_ratio=((0.3, 1), (1, 3.3))): Performs cut-paste augmentation.
        - cutpaste_scar(image, width=[2, 16], length=[10, 25], rotation=(-45, 45)): Performs cut-paste scar augmentation.
        - __call__(image): Calls the cut-paste augmentation based on the specified type.

    References:
        - LilitYolyan/GitHub https://github.com/LilitYolyan/CutPaste/blob/main/cutpaste.py

    """

    def __init__(self, type="binary", data_display_mode=False):
        self.type = type
        self.data_display_mode = data_display_mode
        self.transform = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        )

    def crop_and_paste_patch(self, image, patch_w, patch_h, rotation=False):
        """
        Crops a patch from the image and pastes it back at a random location.

        Parameters:
            - image (PIL.Image.Image): The input image.
            - patch_w (int): Width of the patch.
            - patch_h (int): Height of the patch.
            - rotation (tuple, optional): Range of rotation angles (default is False).

        Returns:
            - PIL.Image.Image: The image with the cropped and pasted patch.
        """
        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(
            0, org_h - patch_h
        )
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))

        # Apply color jitter
        self.transform(patch)

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(
            0, org_h - patch_h
        )
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top), mask=mask)

        if self.data_display_mode:
            # Add a rectangular border based on data_display_mode
            border_color_red = (255, 0, 0)
            border_color_blue = (0, 0, 255)

            border_width = 5
            expansion_size = 10

            draw = ImageDraw.Draw(aug_image)
            draw.rectangle(
                [
                    (paste_left - expansion_size, paste_top - expansion_size),
                    (
                        paste_left + patch_w + expansion_size,
                        paste_top + patch_h + expansion_size,
                    ),
                ],
                outline=border_color_red,
                width=border_width,
            )
            draw.rectangle(
                [
                    (patch_left - expansion_size, patch_top - expansion_size),
                    (
                        patch_left + patch_w + expansion_size,
                        patch_top + patch_h + expansion_size,
                    ),
                ],
                outline=border_color_blue,
                width=border_width,
            )
        return aug_image

    def cutpaste(
        self, image, area_ratio=(0.02, 0.15), aspect_ratio=((0.3, 1), (1, 3.3))
    ):
        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice(
            [random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])]
        )
        patch_w = int(np.sqrt(patch_area * patch_aspect))
        patch_h = int(np.sqrt(patch_area / patch_aspect))
        cutpaste = self.crop_and_paste_patch(image, patch_w, patch_h, rotation=False)
        return cutpaste

    def cutpaste_scar(self, image, width=[2, 16], length=[10, 25], rotation=(-45, 45)):
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar = self.crop_and_paste_patch(
            image, patch_w, patch_h, rotation=rotation
        )
        return cutpaste_scar

    def __call__(self, image):
        if self.type == "binary":
            aug = random.choice([self.cutpaste, self.cutpaste_scar])
            return image, aug(image)

        elif self.type == "3way":
            cutpaste = self.cutpaste(image)
            scar = self.cutpaste_scar(image)
            return image, cutpaste, scar
