import torch
import torch.nn as nn

import timm

from .resnet import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class SingleBaseline(nn.Module):
    def __init__(self, arch="resnet18"):
        super().__init__()

        if arch == "resnet18":
            self.derm_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif arch == "resnet50":
            self.derm_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = 2048

        self.derm_backbone.fc = nn.Identity()

        # Order by: DIAG, PN, BWV, VS, PIG, STR, DaG, RS
        self.classifier = nn.ModuleList(
            [
                nn.Linear(feat_dim, 5),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 2),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 2),
            ]
        )
        for i in self.classifier:
            self._init_fc(i)

    def forward(self, x):
        derm_feats = self.derm_backbone(x)
        feats = derm_feats
        return [classify(feats) for classify in self.classifier]

    def _init_fc(self, layer):
        layer.weight.data.normal_(mean=0.0, std=0.01)
        layer.bias.data.zero_()

    def freeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = False
        for param in self.clinic_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = True
        for param in self.clinic_backbone.parameters():
            param.requires_grad = True


class Baseline(nn.Module):
    def __init__(self, arch="resnet18", weights="IMAGENET1K_V1"):
        super().__init__()

        if arch == "resnet18":
            self.derm_backbone = resnet18(weights=weights)
            self.clinic_backbone = resnet18(weights=weights)
            feat_dim = 512 * 2
        elif arch == "resnet50":
            self.derm_backbone = resnet50(weights=weights)
            self.clinic_backbone = resnet50(weights=weights)
            feat_dim = 2048 * 2
        elif arch.startswith("timm"):
            timm_arch = arch.split("-")[1]
            self.derm_backbone = timm.create_model(timm_arch, pretrained=True)
            self.clinic_backbone = timm.create_model(timm_arch, pretrained=True)
            feat_dim = 2048 * 2

        self.derm_backbone.fc = nn.Identity()
        self.clinic_backbone.fc = nn.Identity()

        # Order by: DIAG, PN, BWV, VS, PIG, STR, DaG, RS
        self.classifier = nn.ModuleList(
            [
                nn.Linear(feat_dim, 5),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 2),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 3),
                nn.Linear(feat_dim, 2),
            ]
        )
        for i in self.classifier:
            self._init_fc(i)

    def forward(self, x):
        derm_feats = self.derm_backbone(x[0])
        clinic_feats = self.clinic_backbone(x[1])
        feats = torch.cat([derm_feats, clinic_feats], dim=1)
        return [classify(feats) for classify in self.classifier]

    def _init_fc(self, layer):
        layer.weight.data.normal_(mean=0.0, std=0.01)
        layer.bias.data.zero_()

    def freeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = False
        for param in self.clinic_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = True
        for param in self.clinic_backbone.parameters():
            param.requires_grad = True


class BaselineMLP1(nn.Module):
    def __init__(self, arch="resnet18"):
        super().__init__()

        if arch == "resnet18":
            self.derm_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.clinic_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512 * 2
        elif arch == "resnet50":
            self.derm_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.clinic_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = 2048 * 2

        self.derm_backbone.fc = nn.Identity()
        self.clinic_backbone.fc = nn.Identity()

        # Order by: DIAG, PN, BWV, VS, PIG, STR, DaG, RS
        self.classifier = nn.ModuleList(
            [
                self._make_fc(feat_dim, 5),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 2),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 2),
            ]
        )
        self._init_fc(self.classifier)

    def forward(self, x):
        derm_feats = self.derm_backbone(x[0])
        clinic_feats = self.clinic_backbone(x[1])
        feats = torch.cat([derm_feats, clinic_feats], dim=1)
        return [classify(feats) for classify in self.classifier]

    def _make_fc(self, feat_dim, num_classes):
        return nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def _init_fc(self, module):
        for i in module:
            if isinstance(i, nn.Linear):
                i.weight.data.normal_(mean=0.0, std=0.01)
                i.bias.data.zero_()

    def freeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = False
        for param in self.clinic_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = True
        for param in self.clinic_backbone.parameters():
            param.requires_grad = True


class BaselineMLP2(nn.Module):
    def __init__(self, arch="resnet18"):
        super().__init__()

        if arch == "resnet18":
            self.derm_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.clinic_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512 * 2
        elif arch == "resnet50":
            self.derm_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.clinic_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = 2048 * 2

        self.derm_backbone.fc = nn.Identity()
        self.clinic_backbone.fc = nn.Identity()

        # Order by: DIAG, PN, BWV, VS, PIG, STR, DaG, RS
        self.classifier = nn.ModuleList(
            [
                self._make_fc(feat_dim, 5),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 2),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 2),
            ]
        )
        self._init_fc(self.classifier)

    def forward(self, x):
        derm_feats = self.derm_backbone(x[0])
        clinic_feats = self.clinic_backbone(x[1])
        feats = torch.cat([derm_feats, clinic_feats], dim=1)
        return [classify(feats) for classify in self.classifier]

    def _make_fc(self, feat_dim, num_classes):
        return nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _init_fc(self, module):
        for i in module:
            if isinstance(i, nn.Linear):
                i.weight.data.normal_(mean=0.0, std=0.01)
                i.bias.data.zero_()

    def freeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = False
        for param in self.clinic_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = True
        for param in self.clinic_backbone.parameters():
            param.requires_grad = True


class BaselineMLP3(nn.Module):
    def __init__(self, arch="resnet18"):
        super().__init__()

        if arch == "resnet18":
            self.derm_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.clinic_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512 * 2
        elif arch == "resnet50":
            self.derm_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.clinic_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = 2048 * 2

        self.derm_backbone.fc = nn.Identity()
        self.clinic_backbone.fc = nn.Identity()

        # Order by: DIAG, PN, BWV, VS, PIG, STR, DaG, RS
        self.classifier = nn.ModuleList(
            [
                self._make_fc(feat_dim, 5),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 2),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 3),
                self._make_fc(feat_dim, 2),
            ]
        )
        self._init_fc(self.classifier)

    def forward(self, x):
        derm_feats = self.derm_backbone(x[0])
        clinic_feats = self.clinic_backbone(x[1])
        feats = torch.cat([derm_feats, clinic_feats], dim=1)
        return [classify(feats) for classify in self.classifier]

    def _make_fc(self, feat_dim, num_classes):
        return nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )

    def _init_fc(self, module):
        for i in module:
            if isinstance(i, nn.Linear):
                i.weight.data.normal_(mean=0.0, std=0.01)
                i.bias.data.zero_()

    def freeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = False
        for param in self.clinic_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.derm_backbone.parameters():
            param.requires_grad = True
        for param in self.clinic_backbone.parameters():
            param.requires_grad = True
