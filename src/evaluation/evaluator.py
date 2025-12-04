"""Model evaluation framework"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.evaluation.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""

    task_name: str
    metrics: Dict[str, float]
    num_samples: int
    batch_size: int
    device: str

    def __repr__(self) -> str:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in self.metrics.items()])
        return f"EvaluationResult({self.task_name}: {metrics_str})"


class ClassificationEvaluator:
    """Evaluator for classification tasks"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        Initialize evaluator

        Args:
            model: Model to evaluate (assumed to already be on correct device)
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        # Model is already on device from load_model(), just set to eval mode
        self.model.eval()

    def evaluate(
        self,
        dataloader: DataLoader,
        task_name: str,
        metrics: List[str],
    ) -> EvaluationResult:
        """
        Evaluate model on a dataset

        Args:
            dataloader: DataLoader for the dataset
            task_name: Name of the task
            metrics: List of metric names to compute

        Returns:
            EvaluationResult object
        """
        logger.info(f"Evaluating {task_name}...")

        # Get predictions
        predictions, labels = self._compute_predictions(dataloader)

        # Compute metrics
        metric_values = compute_classification_metrics(
            predictions=predictions.numpy(),
            labels=labels.numpy(),
            metrics=metrics,
        )

        result = EvaluationResult(
            task_name=task_name,
            metrics=metric_values,
            num_samples=len(labels),
            batch_size=self.batch_size,
            device=self.device,
        )

        logger.info(f"  {result}")
        return result

    def _compute_predictions(self, dataloader: DataLoader) -> tuple:
        """
        Compute model predictions for entire dataset

        Args:
            dataloader: DataLoader for the dataset

        Returns:
            Tuple of (predictions, labels) as tensors
        """
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu())
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return all_preds, all_labels


class MultiTaskEvaluator:
    """Evaluator for multiple tasks"""

    def __init__(
        self,
        evaluator: ClassificationEvaluator,
        task_names: List[str],
        metric_names: List[str],
    ):
        """
        Initialize multi-task evaluator

        Args:
            evaluator: Base evaluator instance
            task_names: List of task names
            metric_names: List of metric names to compute
        """
        self.evaluator = evaluator
        self.task_names = task_names
        self.metric_names = metric_names

    def evaluate_all_tasks(
        self,
        dataloaders: Dict[str, DataLoader],
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate model on all tasks

        Args:
            dataloaders: Dictionary mapping task names to DataLoaders

        Returns:
            Dictionary mapping task names to EvaluationResults
        """
        results = {}

        for task_name in self.task_names:
            logger.info(f"Evaluating task: {task_name}")

            dataloader = dataloaders[task_name]
            result = self.evaluator.evaluate(
                dataloader=dataloader,
                task_name=task_name,
                metrics=self.metric_names,
            )

            results[task_name] = result

        return results
