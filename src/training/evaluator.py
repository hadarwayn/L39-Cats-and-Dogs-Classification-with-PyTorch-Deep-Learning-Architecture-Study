"""
Model evaluation and metrics computation.

After training, the Evaluator tests the model on the validation set
and calculates how well it performed using several metrics:
- Accuracy: what % of images were classified correctly
- Precision: of all images predicted as "cat", how many really were cats
- Recall: of all actual cats, how many did the model find
- F1 Score: a balanced combination of precision and recall
- Confusion Matrix: a 2x2 table showing correct vs wrong predictions
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES


class Evaluator:
    """Evaluates a trained model and computes detailed metrics."""

    @staticmethod
    @torch.no_grad()
    def get_predictions(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the model on all data and collect predictions.

        Returns:
            (all_preds, all_labels, all_probs) as NumPy arrays
        """
        model.eval()
        all_preds: list = []
        all_labels: list = []
        all_probs: list = []

        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

        return (
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs),
        )

    @staticmethod
    def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict:
        """
        Full evaluation: accuracy, precision, recall, F1, confusion matrix.

        Returns a dict with all metrics.
        """
        preds, labels, probs = Evaluator.get_predictions(model, dataloader, device)

        cm = confusion_matrix(labels, preds)
        report = classification_report(
            labels, preds, target_names=CLASS_NAMES, output_dict=True,
        )

        accuracy = float(np.mean(preds == labels))

        return {
            "accuracy": accuracy,
            "precision_cat": report["cat"]["precision"],
            "recall_cat": report["cat"]["recall"],
            "f1_cat": report["cat"]["f1-score"],
            "precision_dog": report["dog"]["precision"],
            "recall_dog": report["dog"]["recall"],
            "f1_dog": report["dog"]["f1-score"],
            "confusion_matrix": cm,
            "predictions": preds,
            "labels": labels,
            "probabilities": probs,
        }

    @staticmethod
    def get_misclassified_samples(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num: int = 6,
    ) -> tuple[list, list[int], list[int]]:
        """
        Find images the model got wrong.

        Returns (images, true_labels, pred_labels) for the first
        `num` misclassified samples.
        """
        model.eval()
        wrong_images: list = []
        wrong_true: list[int] = []
        wrong_pred: list[int] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images_dev = images.to(device)
                outputs = model(images_dev)
                preds = outputs.argmax(dim=1).cpu()

                # Find wrong predictions in this batch
                mask = preds != labels
                for idx in mask.nonzero(as_tuple=True)[0]:
                    i = idx.item()
                    wrong_images.append(images[i])
                    wrong_true.append(labels[i].item())
                    wrong_pred.append(preds[i].item())
                    if len(wrong_images) >= num:
                        return wrong_images, wrong_true, wrong_pred

        return wrong_images, wrong_true, wrong_pred
