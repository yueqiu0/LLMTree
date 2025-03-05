import numpy as np


class LossFunction:
    def __call__(self, y_pred, y_llm, y_tree):
        raise NotImplementedError


class MyLossFunction(LossFunction):
    def __init__(self, num_classes: int, lambda_: float, mu: float) -> None:
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.mu = mu

    def __call__(
        self, y_true: np.ndarray, y_llm: np.ndarray, y_tree: np.ndarray
    ) -> tuple[float, list[float]]:
        # llm loss: number of misclassified examples
        llm_loss = np.sum(y_true != y_llm)
        # tree loss: (1) number of misclassified examples of known cases (>= 0),
        # (2) mu * number of unknown examples (-1)
        tree_loss = self.mu * np.sum(y_tree == -1) + np.sum(
            y_true != y_tree, where=y_tree >= 0
        )

        return llm_loss + self.lambda_ * tree_loss, [llm_loss, tree_loss]
