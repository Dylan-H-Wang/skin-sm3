import torch
import torch.nn as nn

import resnet



CLASSES_NAME = ["DIAG", "PN", "BWV", "VS", "PIG", "STR", "DaG", "RS"]
NUM_CLASSES = [5, 3, 2, 3, 3, 3, 3, 2]
CLS_WEIGHTS = [2, 2, 1, 2, 2, 2, 2, 1]
CLASSES_NAME_2 = [
    f"{CLASSES_NAME[i]}-{j+1}" for i in range(len(CLASSES_NAME)) for j in range(NUM_CLASSES[i])
]


class MultiLabelProjector(nn.Module):
    def __init__(self, in_dim, proj_dim, num_labels):
        super().__init__()
        self.projectors = nn.ModuleList(
            [self._make_projector(in_dim, proj_dim) for _ in range(num_labels)]
        )

    def _make_projector(self, in_dim, proj_dim):
        return nn.Sequential(
            nn.Linear(in_dim, proj_dim),
        )  # output layer

    def forward(self, x):
        return [projector(x) for projector in self.projectors]
    

class Extractor(nn.Module):
    def __init__(self, arch, weights=None) -> None:
        super().__init__()

        self.derm_backbone = resnet.__dict__[arch](weights=weights)
        self.derm_feat_dim = self.derm_backbone.fc.in_features
        self.derm_backbone.fc = nn.Identity()

        self.clinic_backbone = resnet.__dict__[arch](weights=weights)
        self.clinic_feat_dim = self.clinic_backbone.fc.in_features
        self.clinic_backbone.fc = nn.Identity()

    def forward(self):
        pass

    def extract(self, derm_imgs, clinic_imgs):
        derm_feat = self.derm_backbone(derm_imgs)
        clinic_feat = self.clinic_backbone(clinic_imgs)
        return [derm_feat, clinic_feat]
    

class Model(nn.Module):
    def __init__(self, extractor, projectors, feat_dim, l2_norm, n_heads, sa_dim_ff, sa_dropout):
        super().__init__()
        self.extractor = extractor
        self.projectors = projectors
        self.mlc_sa = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=sa_dim_ff, dropout=sa_dropout
        )
        self.feat_dim = feat_dim
        self.l2_norm = l2_norm
        self.prototypes = self._make_prototype()

    def _make_prototype(self):
        prototypes = nn.ModuleList([nn.Linear(self.feat_dim, i) for i in NUM_CLASSES])

        for i in prototypes:
            self._init_fc(i)

        return prototypes

    def _init_fc(self, layer):
        layer.weight.data.normal_(mean=0.0, std=0.01)
        layer.bias.data.zero_()

    def forward(self, derm_imgs, clinic_imgs):
        feats = self.extractor.extract(derm_imgs, clinic_imgs)
        feats = torch.cat(feats, dim=1)

        proj_feats = self.projectors(feats)
        if not isinstance(proj_feats, list):
            proj_feats = [proj_feats]

        proj_feats = torch.stack(proj_feats, dim=0)
        sa_feats = self.mlc_sa(proj_feats)

        if self.l2_norm:
            for i in range(len(sa_feats)):
                sa_feats[i] = nn.functional.normalize(sa_feats[i], dim=-1, p=2)

        preds = [
            self.prototypes[i](sa_feats[i % len(sa_feats)]) for i in range(len(self.prototypes))
        ]

        return preds
    
if __name__ == "__main__":
    arch = "resnet50"
    mlc_proj_dim = 512
    num_labels = 8
    l2_norm = False
    num_heads = 1
    sa_dim_ff = 128
    sa_dropout = 0.1
    # pretrain_path = "./best_linear.pth"
    pretrain_path = "./best_finetune.pth"

    extractor = Extractor(arch)
    feat_dim = extractor.derm_feat_dim + extractor.clinic_feat_dim
    projectors = MultiLabelProjector(feat_dim, mlc_proj_dim, num_labels)
    evaluator = Model(
        extractor,
        projectors,
        mlc_proj_dim,
        l2_norm,
        num_heads,
        sa_dim_ff,
        sa_dropout,
    )

    print(f"Loading pre-trained weights from '{pretrain_path}' ...")
    state_dict = torch.load(pretrain_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        if "encoder." in k:
            state_dict[k.replace("encoder.", "")] = state_dict.pop(k)
    msg = evaluator.load_state_dict(state_dict, strict=True)
    print(f"loaded pre-trained model weights from '{pretrain_path}'")
    
    evaluator = evaluator.cuda()
    print(evaluator)

    #todo: your evaluation code here
    dummy_derm_imgs = torch.randn(1, 3, 224, 224).cuda()
    dummy_clinic_imgs = torch.randn(1, 3, 224, 224).cuda()
    evaluator(dummy_derm_imgs, dummy_clinic_imgs)