# Modified from https://github.com/facebookresearch/SimCLR
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from src.models import resnet


def make_projector(in_dim, proj_dim):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim, bias=False),
        nn.BatchNorm1d(in_dim),
        nn.ReLU(inplace=True),  # first layer
        nn.Linear(in_dim, in_dim, bias=False),
        nn.BatchNorm1d(in_dim),
        nn.ReLU(inplace=True),  # second layer
        nn.Linear(in_dim, proj_dim, bias=False),
        nn.BatchNorm1d(proj_dim, affine=False),
    )  # output layer


# main class
class SimCLR(nn.Module):
    def __init__(
        self,
        arch,
        weights=None,
        proj_dim=128,
        temperature=0.5,
        return_feats=False,
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.temperature = temperature
        self.return_feats = return_feats

        # define encoders
        self.encoder = resnet.__dict__[arch](weights=weights)
        self.encoder_out_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # define projector
        self.projector = make_projector(self.encoder_out_dim, self.proj_dim)

    def forward(self, x1, x2):
        bs = x1.shape[0]

        # compute features for one view
        f1 = self.encoder(x1)  # NxC
        f2 = self.encoder(x2)  # NxC

        features = self.projector(torch.cat([f1, f2], dim=0))
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(x1.device)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(x1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x1.device)

        logits = logits / self.temperature

        if self.return_feats:
            return (logits, labels), (f1, f2)
        else:
            return (logits, labels)

    def extract(self, imgs):
        return self.encoder(imgs)


class SimCLRSkin(nn.Module):
    def __init__(self, arch, weights=None, proj_dim=128, temperature=0.5) -> None:
        super().__init__()

        self.derm_backbone = SimCLR(arch, weights, proj_dim, temperature)
        self.clinic_backbone = SimCLR(arch, weights, proj_dim, temperature)

    def forward(self, derm_imgs, clinic_imgs):
        derm_outs = self.derm_backbone(*derm_imgs)
        clinic_outs = self.clinic_backbone(*clinic_imgs)

        return (derm_outs, clinic_outs)

    def extract(self, derm_imgs, clinic_imgs):
        derm_feat = self.derm_backbone.encoder(derm_imgs)
        clinic_feat = self.clinic_backbone.encoder(clinic_imgs)
        return [derm_feat, clinic_feat]


class SimCLRSkinV2(nn.Module):
    # concat dermoscopy images and clinic images
    # contrast the combined features too
    def __init__(self, arch, weights=None, proj_dim=128, temperature=0.5) -> None:
        super().__init__()
        self.temperature = temperature

        self.derm_backbone = SimCLR(arch, weights, proj_dim, temperature, True)
        self.clinic_backbone = SimCLR(arch, weights, proj_dim, temperature, True)

        derm_feat_dim = self.derm_backbone.encoder_out_dim
        clinic_feat_dim = self.clinic_backbone.encoder_out_dim
        cross_feat_dim = derm_feat_dim + clinic_feat_dim

        self.cross_proj = make_projector(cross_feat_dim, proj_dim)

    def _cal_logits(self, f1, f2, projector, temperature):
        bs = f1.shape[0]

        features = projector(torch.cat([f1, f2], dim=0))
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(f1.device)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(f1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(f1.device)

        logits = logits / temperature

        return (logits, labels)

    def forward(self, derm_imgs, clinic_imgs):
        derm_outs, derm_feats = self.derm_backbone(*derm_imgs)
        clinic_outs, clinic_feats = self.clinic_backbone(*clinic_imgs)

        cross_feats_1 = torch.cat([derm_feats[0], clinic_feats[0]], dim=1)
        cross_feats_2 = torch.cat([derm_feats[1], clinic_feats[1]], dim=1)

        cross_outs = self._cal_logits(
            cross_feats_1, cross_feats_2, self.cross_proj, self.temperature
        )

        return (derm_outs, clinic_outs, cross_outs)

    def extract(self, derm_imgs, clinic_imgs):
        derm_feat = self.derm_backbone.encoder(derm_imgs)
        clinic_feat = self.clinic_backbone.encoder(clinic_imgs)
        return [derm_feat, clinic_feat]


class SimCLRSkinV21(SimCLRSkinV2):
    # concat dermoscopy images and clinic images in a cross way
    def __init__(self, arch, weights=None, proj_dim=128, temperature=0.5) -> None:
        super().__init__(arch, weights, proj_dim, temperature)

    def forward(self, derm_imgs, clinic_imgs):
        derm_outs, derm_feats = self.derm_backbone(*derm_imgs)
        clinic_outs, clinic_feats = self.clinic_backbone(*clinic_imgs)

        cross_feats_1 = torch.cat([derm_feats[0], clinic_feats[1]], dim=1)
        cross_feats_2 = torch.cat([derm_feats[1], clinic_feats[0]], dim=1)

        cross_outs = self._cal_logits(
            cross_feats_1, cross_feats_2, self.cross_proj, self.temperature
        )

        return (derm_outs, clinic_outs, cross_outs)


class SimCLRSkinV22(SimCLRSkinV2):
    # concat dermoscopy images and clinic images in a normal and cross way
    def __init__(self, arch, weights=None, proj_dim=128, temperature=0.5) -> None:
        super().__init__(arch, weights, proj_dim, temperature)

    def forward(self, derm_imgs, clinic_imgs):
        derm_outs, derm_feats = self.derm_backbone(*derm_imgs)
        clinic_outs, clinic_feats = self.clinic_backbone(*clinic_imgs)

        cross_feats_1 = torch.cat([derm_feats[0], clinic_feats[0]], dim=1)
        cross_feats_2 = torch.cat([derm_feats[1], clinic_feats[1]], dim=1)
        cross_feats_3 = torch.cat([derm_feats[0], clinic_feats[1]], dim=1)
        cross_feats_4 = torch.cat([derm_feats[1], clinic_feats[0]], dim=1)

        cross_out_1 = self._cal_logits(
            cross_feats_1, cross_feats_2, self.cross_proj, self.temperature
        )
        cross_out_2 = self._cal_logits(
            cross_feats_3, cross_feats_4, self.cross_proj, self.temperature
        )
        cross_outs = (cross_out_1, cross_out_2)

        return (derm_outs, clinic_outs, cross_outs)


class SimCLRSkinV23(SimCLRSkinV2):
    # add dermoscopy images and clinic images and contrast the combined features
    def __init__(self, arch, weights=None, proj_dim=128, temperature=0.5) -> None:
        super().__init__(arch, weights, proj_dim, temperature)

        self.cross_proj = make_projector(self.derm_backbone.encoder_out_dim, proj_dim)

    def forward(self, derm_imgs, clinic_imgs):
        derm_outs, derm_feats = self.derm_backbone(*derm_imgs)
        clinic_outs, clinic_feats = self.clinic_backbone(*clinic_imgs)

        cross_feats = [derm_feats[0] + clinic_feats[0], derm_feats[1] + clinic_feats[1]]
        cross_outs = self._cal_logits(
            cross_feats[0], cross_feats[1], self.cross_proj, self.temperature
        )

        return (derm_outs, clinic_outs, cross_outs)
    

class SimCLRSkinV3(nn.Module):
    # instead of concat, contrast derm and clinic images
    # share the same projector
    def __init__(self, arch, weights=None, proj_dim=128, temperature=0.5, use_checkpoint=False) -> None:
        super().__init__()
        self.temperature = temperature

        self.derm_backbone = SimCLR(arch, weights, proj_dim, temperature, True)
        self.clinic_backbone = SimCLR(arch, weights, proj_dim, temperature, True)

        self.derm_feat_dim = self.derm_backbone.encoder_out_dim
        self.clinic_feat_dim = self.clinic_backbone.encoder_out_dim
        self.cross_feat_dim = self.derm_feat_dim

        self.cross_proj = make_projector(self.cross_feat_dim, proj_dim)

        if use_checkpoint:
            self._apply_checkpoint()

    def _apply_checkpoint(self):
        fisrt_layer_1 = deepcopy(self.derm_backbone.encoder.conv1)
        fisrt_layer_2 = deepcopy(self.clinic_backbone.encoder.conv1)

        checkpoint_impl = CheckpointImpl.NO_REENTRANT
        checkpoint_wrapper_fn = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=checkpoint_impl,
        )
        check_fn = lambda submodule: isinstance(submodule, (nn.Conv2d, nn.Linear))
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=check_fn,
        )

        # disable checkpointing for the first conv layer
        self.derm_backbone.encoder.conv1 = fisrt_layer_1
        self.clinic_backbone.encoder.conv1 = fisrt_layer_2

    def _cal_logits(self, f1, f2, projector1, projector2, temperature):
        bs = f1.shape[0]

        features = torch.cat([projector1(f1), projector2(f2)], dim=0)
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(f1.device)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(f1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(f1.device)

        logits = logits / temperature

        return (logits, labels)

    def forward(self, derm_imgs, clinic_imgs, style):
        derm_outs, derm_feats = self.derm_backbone(*derm_imgs)
        clinic_outs, clinic_feats = self.clinic_backbone(*clinic_imgs)

        if style == 0:
            cross_out_1 = self._cal_logits(
                derm_feats[0],
                clinic_feats[0],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_out_2 = self._cal_logits(
                derm_feats[1],
                clinic_feats[1],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_outs = (cross_out_1, cross_out_2)
        elif style == 1:
            cross_out_1 = self._cal_logits(
                derm_feats[0],
                clinic_feats[1],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_out_2 = self._cal_logits(
                derm_feats[1],
                clinic_feats[0],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_outs = (cross_out_1, cross_out_2)
        elif style == 2:
            cross_out_1 = self._cal_logits(
                derm_feats[0],
                clinic_feats[0],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_out_2 = self._cal_logits(
                derm_feats[0],
                clinic_feats[1],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_out_3 = self._cal_logits(
                derm_feats[1],
                clinic_feats[0],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_out_4 = self._cal_logits(
                derm_feats[1],
                clinic_feats[1],
                self.cross_proj,
                self.cross_proj,
                self.temperature,
            )
            cross_outs = (cross_out_1, cross_out_2, cross_out_3, cross_out_4)

        return (derm_outs, clinic_outs, cross_outs)

    def extract(self, derm_imgs, clinic_imgs):
        derm_feat = self.derm_backbone.encoder(derm_imgs)
        clinic_feat = self.clinic_backbone.encoder(clinic_imgs)
        return [derm_feat, clinic_feat]


class SimCLRSkinV32(SimCLRSkinV3):
    # instead of concat, contrast derm and clinic images
    # have independent projector
    def __init__(self, arch, weights=None, proj_dim=128, temperature=0.5, use_checkpoint=False) -> None:
        super().__init__(arch, weights, proj_dim, temperature)

        self.cross_proj = nn.ModuleList(
            [
                make_projector(self.derm_feat_dim, proj_dim),
                make_projector(self.clinic_feat_dim, proj_dim),
            ]
        )

        if use_checkpoint:
            self._apply_checkpoint()

    def forward(self, derm_imgs, clinic_imgs, style):
        derm_outs, derm_feats = self.derm_backbone(*derm_imgs)
        clinic_outs, clinic_feats = self.clinic_backbone(*clinic_imgs)

        if style == 0:
            cross_out_1 = self._cal_logits(
                derm_feats[0],
                clinic_feats[0],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_out_2 = self._cal_logits(
                derm_feats[1],
                clinic_feats[1],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_outs = (cross_out_1, cross_out_2)
        elif style == 1:
            cross_out_1 = self._cal_logits(
                derm_feats[0],
                clinic_feats[1],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_out_2 = self._cal_logits(
                derm_feats[1],
                clinic_feats[0],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_outs = (cross_out_1, cross_out_2)
        elif style == 2:
            cross_out_1 = self._cal_logits(
                derm_feats[0],
                clinic_feats[0],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_out_2 = self._cal_logits(
                derm_feats[0],
                clinic_feats[1],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_out_3 = self._cal_logits(
                derm_feats[1],
                clinic_feats[0],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_out_4 = self._cal_logits(
                derm_feats[1],
                clinic_feats[1],
                self.cross_proj[0],
                self.cross_proj[1],
                self.temperature,
            )
            cross_outs = (cross_out_1, cross_out_2, cross_out_3, cross_out_4)

        return (derm_outs, clinic_outs, cross_outs)
