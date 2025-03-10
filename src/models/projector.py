
from torch import nn


class MultiLabelProjector(nn.Module):
    def __init__(self, in_dim, proj_dim, num_labels):
        super().__init__()
        self.projectors = nn.ModuleList(
            [self._make_projector(in_dim, proj_dim) for _ in range(num_labels)]
        )

    def _make_projector(self, in_dim, proj_dim):
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

    def forward(self, x):
        return [projector(x) for projector in self.projectors]


class MultiLabelProjector2(nn.Module):
    def __init__(self, in_dim, proj_dim, num_labels):
        super().__init__()
        self.projectors = nn.ModuleList(
            [self._make_projector(in_dim, proj_dim) for _ in range(num_labels)]
        )

    def _make_projector(self, in_dim, proj_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )  # output layer

    def forward(self, x):
        return [projector(x) for projector in self.projectors]


class MultiLabelProjector3(nn.Module):
    def __init__(self, in_dim, proj_dim, num_labels):
        super().__init__()
        self.projectors = nn.ModuleList(
            [self._make_projector(in_dim, proj_dim) for _ in range(num_labels)]
        )

    def _make_projector(self, in_dim, proj_dim):
        return nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )  # output layer

    def forward(self, x):
        return [projector(x) for projector in self.projectors]


class MultiLabelProjector4(nn.Module):
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