import torch
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc, average_precision_score
from torchmetrics.classification import Dice
import matplotlib.pyplot as plt

from pathlib import Path


class Evaluator:
    def __init__(
        self,
        model,
        device,
        pathology,
        test_data_loader,
        output_dir="./results",
    ):
        self.model = model
        self.device = device
        self.pathology = pathology
        self.output_dir = output_dir
        self.test_data_loader = test_data_loader
        self.gde = KernelDensity(kernel="gaussian", bandwidth=1)

    def evaluate(self):
        labels = []
        embeds = []
        with torch.no_grad():
            for data, label in self.test_data_loader:
                embed, _ = self.model(data)
                embeds.append(embed.cpu())
                labels.append(label.cpu())
                del embed, _

            labels = torch.cat(labels)
            embeds = torch.cat(embeds)

            self.gde.fit(self.model.embeds.cpu())
            scores = self._predict(embeds)

            dice = Dice(average="micro")
            dice_score = dice(torch.from_numpy(scores), labels)
            print(f"Dice score: {dice_score:.3f}")

            ap_score = average_precision_score(labels, torch.from_numpy(scores))
            print(f"Average precision: {ap_score:.3f}")

            fpr, tpr, roc_auc = self.calculate_roc(scores, labels)
            self.plot_roc(fpr, tpr, roc_auc, self.pathology, self.output_dir)
            return roc_auc, dice_score, ap_score

    def _predict(self, embeddings):
        scores = self.gde.score_samples(embeddings)
        scores = -scores
        return scores

    def calculate_roc(self, scores, labels):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        print(f"ROC AUC: {roc_auc:.3f}")
        return fpr, tpr, roc_auc

    def plot_roc(self, fpr, tpr, roc_auc, pathology, save_path):
        plt.title(f"ROC curve: {pathology}")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")

        image_path = Path(save_path) / f"roc_{pathology}.png"
        plt.savefig(image_path)
        plt.close()
