import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torchvision.models import resnet18, ResNet18_Weights


class CutPasteNet(pl.LightningModule):
    def __init__(self, config, head_layer_count=2, num_classes=2):
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

        head = nn.Sequential(*projection_layers)
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        logits = self.out(self.head(embeds))
        return embeds, logits
