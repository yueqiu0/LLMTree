from .strategy import TrainStrategy, FeatureBaggingStrategy
from .. import logger


class Classifier:
    def __init__(self, strategy: TrainStrategy) -> None:
        self.strategy = strategy

    def fit(self, x, y) -> float:
        logger.log("Train data:")
        for xx, yy in zip(x.tolist(), y.tolist()):
            logger.log(f"{xx} -> {yy}")
        self.strategy.set_train_data(x, y)
        step = 0
        result = (True, None)
        last_loss = None
        while result[0]:
            result = self.strategy.step()
            if result[1] != None:
                last_loss = result[1]
            logger.log(f"Step {step}: loss={result[1]}")
            logger.log("Tree:\n" + self.strategy.get_tree().to_graphviz_source())
            step += 1

        return last_loss

    def predict(self, x) -> tuple[list[int], list[int], list[int]]:
        if type(self.strategy) == FeatureBaggingStrategy:
            sub_results = self.strategy.predict_llm_with_all_subtrees(
                x, with_examples=True
            )
        else:
            sub_results = []

        return (
            self.strategy.predict_llm_with_tree(x, with_examples=True),
            self.strategy.predict_tree(x),
            self.strategy.predict_tree_raw(x),
            sub_results,
        )

    def export(self) -> dict:
        """ """
        return self.strategy.export()

    def to_graphviz(self, features: list[str], classes: list[str]):
        return self.strategy.get_tree().to_graphviz(features, classes)
