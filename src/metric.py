from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        # 클래스별 True Positive, False Positive, False Negative 상태 누적
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: (B x C) tensor, target: (B,) tensor
        pred_labels = torch.argmax(preds, dim=1)
        assert target.ndim == 1 and pred_labels.shape == target.shape, "Shape mismatch"
        for c in range(self.num_classes):
            tp = ((pred_labels == c) & (target == c)).sum().item()
            fp = ((pred_labels == c) & (target != c)).sum().item()
            fn = ((pred_labels != c) & (target == c)).sum().item()

            self.true_positives[c] += tp
            self.false_positives[c] += fp
            self.false_negatives[c] += fn

    def compute(self):
        eps = 1e-8
        precision = self.true_positives / (self.true_positives + self.false_positives + eps)
        recall = self.true_positives / (self.true_positives + self.false_negatives + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        # 모든 카테고리의 F1 점수 평균
        return f1.mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        pred_labels = torch.argmax(preds, dim=1)


        # [TODO] check if preds and target have equal shape
        assert preds.shape[0] == target.shape[0], "preds and target must have the same batch size"

        # [TODO] Cound the number of correct prediction
        correct = (pred_labels == target).sum()
        
        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct.float() / self.total.float()
