import csv
import os

import numpy as np
from sentence_transformers.evaluation import SentenceEvaluator


class PrecomputedCosineEvaluator(SentenceEvaluator):
    def __init__(
        self,
        sentences,
        target_embeddings: np.ndarray,
        name="dev_cosine",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ) -> None:
        """Evaluator that computes cosine similarity against precomputed target embeddings."""
        super().__init__()
        self.sentences = list(sentences)
        self.targets = np.asarray(target_embeddings)  # shape: [N, D]
        self.targets /= np.linalg.norm(self.targets, axis=1, keepdims=True)
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.primary_metric = f"{name}_mean"  # trainer can select-best on this

        assert len(self.sentences) == len(self.targets), "texts and embeddings length mismatch"

    def __call__(self, model, output_path=None, epoch: int = -1, steps: int = -1) -> dict[str, float]:
        """Evaluate the model."""
        # Encode with the current student
        preds = model.encode(
            self.sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )

        if preds.shape[1] != self.targets.shape[1]:
            raise ValueError(f"Dim mismatch: student {preds.shape[1]} vs targets {self.targets.shape[1]}")

        preds = preds / np.linalg.norm(preds, axis=1, keepdims=True)
        cosine = 1 - (preds * self.targets).sum(axis=1).mean()
        scores = {self.primary_metric: cosine}

        if output_path and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, f"{self.name}_results.csv")
            write_header = not os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["epoch", "steps", self.primary_metric])
                w.writerow([epoch, steps, cosine])

        return scores
