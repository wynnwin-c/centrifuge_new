from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, feat_dim: int = 256):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(c, 2 * c, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(2 * c),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * c, 2 * c, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2 * c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(2 * c, 4 * c, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4 * c),
            nn.ReLU(inplace=True),
            nn.Conv1d(4 * c, 4 * c, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(4 * c),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * c, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class FewShotBinaryNet(nn.Module):
    def __init__(self, in_channels: int = 1, feat_dim: int = 256, emb_dim: int = 128):
        super().__init__()
        self.backbone = Conv1DBackbone(in_channels=in_channels, feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 2),
        )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, emb_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encode(x)
        emb = F.normalize(self.projector(feat), dim=1)
        logits = self.classifier(feat)
        return {"feat": feat, "emb": emb, "logits": logits}

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["logits"]


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if embeddings.size(0) < 2:
        return embeddings.new_tensor(0.0)

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives = mask.sum(dim=1)
    valid = positives > 0
    if not torch.any(valid):
        return embeddings.new_tensor(0.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives.clamp_min(1.0)
    loss = -mean_log_prob_pos[valid].mean()
    return loss
