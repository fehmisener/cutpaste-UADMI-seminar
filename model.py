import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torchvision.models import resnet18, ResNet18_Weights


class CutPasteNet(pl.LightningModule):
    """
    CutPasteNet is a PyTorch Lightning module for image classification tasks,
    specifically designed to work with the Cut-Paste method of anomaly detection.
    It inherits from pl.LightningModule and encapsulates a ResNet18-based model
    with customizable head layers and class count. The implementation is based on
    the principles outlined in the paper "CutPaste: Self-Supervised Learning for Anomaly
    Detection and Localization"

    Attributes:
        config (dict): A configuration dictionary containing training parameters.
        model (_RestNet18): An instance of a ResNet18-based model.
        criterion (torch.nn.Module): The loss function (CrossEntropyLoss).
        embeds (list): A list to store embeddings (empty initially).

    Args:
        config (dict): Configuration settings, including learning rate and momentum.
        head_layer_count (int): The number of head layers in the ResNet model. Default is 2.
        num_classes (int): The number of classes for classification. Default is 2.

    Methods:
        configure_optimizers: Sets up the optimizer for training.
        forward: Defines the forward pass of the model.
        training_step: Implements a single training step.
        on_train_end: Actions to perform at the end of each training epoch.

    References:
        - CutPaste: Self-Supervised Learning for Anomaly Detection and Localization
          https://arxiv.org/abs/2104.04015
        - PyTorch Lightning
          https://pytorch-lightning.readthedocs.io/en/latest/
        - ResNet18
          https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18
    """

    def __init__(self, config, head_layer_count=2, num_classes=2):
        """
        Initializes the CutPasteNet model with the given configuration.

        Args:
            config (dict): Configuration settings, including learning rate and momentum.
            head_layer_count (int, optional): The number of head layers in the ResNet model. Defaults to 2.
            num_classes (int, optional): The number of classes for classification. Defaults to 2.

        Returns:
            None
        """
        super().__init__()

        if torch.backends.mps.is_available():
            self.deviceType = torch.device("mps")
        else:
            self.deviceType = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print("Selected device: %s" % self.deviceType)

        self.config = config
        self.model = _RestNet18(
            head_layer_count=head_layer_count, num_classes=num_classes
        ).to(self.deviceType)
        self.criterion = nn.CrossEntropyLoss()
        self.embeds = []

    def configure_optimizers(self):
        """
        Sets up the optimizer for the model. This method is used by PyTorch Lightning
        to get the optimizers and learning rate schedulers.

        Args:
            None

        Returns:
            The configured optimizer (and optionally, the learning rate scheduler).
        """
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, self.config["num_epochs"]
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing the data.

        Returns:
            The output of the model given the input x.
        """
        embeds, logits = self.model(x)
        return embeds, logits

    def training_step(self, batch):
        # Forward pass
        x = torch.cat(batch, axis=0)
        embeds, logits = self.model(x)

        # Loss computation
        y = torch.arange(len(batch), device=self.deviceType)
        y = y.repeat_interleave(batch[0].size(0))
        loss = self.criterion(logits, y)

        # Metric computation
        predicted = torch.argmax(logits, axis=1)
        accuracy = torch.true_divide(torch.sum(predicted == y), predicted.size(0))

        # Logging
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)

        self.embeds.append(embeds.to(self.deviceType))
        return loss

    def on_train_end(self):
        self.embeds = torch.cat(self.embeds)


class _RestNet18(nn.Module):
    def __init__(self, head_layer_count=2, num_classes=3):
        super(_RestNet18, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        last_layer = 512
        projection_layers = []
        for num_neurons in [512] * head_layer_count + [128]:
            projection_layers.append(nn.Linear(last_layer, num_neurons))
            projection_layers.append(nn.BatchNorm1d(num_neurons))
            projection_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        self.resnet18.fc = nn.Identity()
        self.head = nn.Sequential(*projection_layers)
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        logits = self.out(self.head(embeds))
        return embeds, logits
