import torch
import torch.nn as nn
from torch.nn import functional as F


# def concat_all_gather(tensor: torch.Tensor, accelerator: Accelerator):
#     return accelerator.all_gather(tensor).view(-1, *tensor.shape[1:])


# modified from https://github.com/Lightning-AI/lightning-bolts/pull/725
class KNNOnlineEvaluator(nn.Module):
    """Weighted KNN online evaluator for self-supervised learning.
    The weighted KNN classifier matches sec 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
    The implementation follows:
        1. https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
        2. https://github.com/leftthomas/SimCLR
        3. https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    Example::
        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)
        online_eval = KNNOnlineEvaluator(
            k=100,
            temperature=0.1
        )
    """

    def __init__(
        self, train_dataloader, val_dataloader, n_classes, k=200, temperature=0.07
    ) -> None:
        """
        Args:
            k: k for k nearest neighbor
            temperature: temperature. See tau in section 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_classes = n_classes
        self.k = k
        self.temperature = temperature

    def predict(
        self,
        query_feature: torch.Tensor,
        feature_bank: torch.Tensor,
        target_bank: torch.Tensor,
    ):
        """
        Args:
            query_feature: (B, D) a batch of B query vectors with dim=D
            feature_bank: (N, D) the bank of N known vectors with dim=D
            target_bank: (N, ) the bank of N known vectors' labels
        Returns:
            (B, ) the predicted labels of B query vectors
        """

        B = query_feature.shape[0]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = query_feature @ feature_bank.T
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(target_bank.expand(B, -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.temperature).exp()

        # counts for each class
        one_hot_label = torch.zeros(
            B * self.k, self.num_classes, device=sim_labels.device
        )
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )
        # weighted score ---> [B, C]
        pred_scores = torch.sum(
            one_hot_label.view(B, -1, self.num_classes) * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    @torch.inference_mode()
    def on_validation_epoch_end(self, model) -> None:
        model.eval()

        total_top1, total_num, feature_bank, target_bank = 0.0, 0, [], []

        # go through train data to generate feature bank
        for inputs, labels in self.train_dataloader:
            x = inputs.cuda(non_blocking=True)
            target = labels.cuda(non_blocking=True)

            feature = model(x).flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            feature_bank.append(feature)
            target_bank.append(target)

        # [N, D]
        feature_bank = torch.cat(feature_bank, dim=0)
        # [N]
        target_bank = torch.cat(target_bank, dim=0)

        # go through val data to predict the label by weighted knn search
        for inputs, labels in self.val_dataloader:
            x = inputs.cuda(non_blocking=True)
            target = labels.cuda(non_blocking=True)

            feature = model(x).flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            pred_labels = self.predict(feature, feature_bank, target_bank)

            total_num += x.shape[0]
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

        return total_top1 / total_num


class LogisticRegressEvaluator(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()

        self.classifier = nn.Linear(feat_dim, n_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        return self.classifier(x)


class LogisticRegressMultiHeadEvaluator(nn.Module):
    def __init__(self, feat_dim, n_classes_per_label):
        super().__init__()

        self.classifier = nn.ModuleList(
            [nn.Linear(feat_dim, i) for i in n_classes_per_label]
        )
        for head in self.classifier:
            head.weight.data.normal_(mean=0.0, std=0.01)
            head.bias.data.zero_()

    def forward(self, x):
        return [classify(x) for classify in self.classifier]
