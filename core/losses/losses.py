"""
Loss functions for MoE Fusion Segmentor.

  1. WeightedCE     — cross-entropy with per-class frequency reweighting + label smoothing
  2. LovaszSoftmax  — directly optimizes per-class Jaccard (mIoU proxy)
  3. GateEntropy    — penalizes gate collapse by maximizing entropy of gate weights
  4. MoEFusionLoss  — combines all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Lovász-Softmax (adapted from Berman et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probas(probas, labels, ignore_index):
    """Flatten predictions and labels, remove ignored pixels."""
    # probas: (B, C, H, W)  labels: (B, H, W)
    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)
    labels = labels.reshape(-1)  # (BHW,)
    if ignore_index is not None:
        valid = labels != ignore_index
        probas = probas[valid]
        labels = labels[valid]
    return probas, labels


def _lovasz_softmax_flat(probas, labels, num_classes):
    """Multi-class Lovász-Softmax loss on flattened predictions."""
    losses = []
    for c in range(num_classes):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.detach()
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    """Lovász-Softmax loss for multi-class segmentation."""

    def __init__(self, ignore_index=0, num_classes=6):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, H, W)
            labels: (B, H, W) with class indices
        """
        probas = F.softmax(logits, dim=1)
        probas, labels = _flatten_probas(probas, labels, self.ignore_index)
        return _lovasz_softmax_flat(probas, labels, self.num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Cross-Entropy with label smoothing
# ─────────────────────────────────────────────────────────────────────────────

class WeightedCE(nn.Module):
    """Cross-entropy with optional per-class weights and label smoothing."""

    def __init__(self, ignore_index=0, label_smoothing=0.1,
                 class_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer('weight',
                                 torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits, labels):
        return F.cross_entropy(
            logits, labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gate Entropy Regularization
# ─────────────────────────────────────────────────────────────────────────────

class GateEntropyLoss(nn.Module):
    """Penalizes gate collapse by maximizing entropy of gate weight distribution.

    Loss = -mean(entropy) over the batch.
    Lower entropy = more collapsed gate = higher loss.

    Gate weights are expected to be softmax * num_scales, so we
    renormalize to a proper distribution before computing entropy.
    """

    def __init__(self, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, gate_weights):
        """
        Args:
            gate_weights: (B, num_scales) — from ScaleGate (softmax * num_scales)
        """
        # Renormalize to proper distribution
        probs = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (B,)
        # Negative entropy: minimize this to maximize entropy
        return -entropy.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class MoEFusionLoss(nn.Module):
    """Combined loss: CE + Lovász + Gate Entropy.

    L_total = ce_weight * L_CE + lovasz_weight * L_Lovász
              + gate_entropy_weight * L_gate_entropy
    """

    def __init__(self, num_classes=6, ignore_index=0,
                 ce_weight=1.0, lovasz_weight=1.0,
                 gate_entropy_weight=0.1, label_smoothing=0.1,
                 class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.gate_entropy_weight = gate_entropy_weight

        self.ce = WeightedCE(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )
        self.lovasz = LovaszSoftmax(
            ignore_index=ignore_index,
            num_classes=num_classes,
        )
        self.gate_entropy = GateEntropyLoss(num_scales=4)

    def forward(self, logits, labels, gate_weights):
        """
        Args:
            logits: (B, C, H, W)
            labels: (B, H, W)
            gate_weights: (B, 4)
        Returns:
            dict with 'total', 'ce', 'lovasz', 'gate_entropy' losses
        """
        l_ce = self.ce(logits, labels)
        l_lovasz = self.lovasz(logits, labels)
        l_gate = self.gate_entropy(gate_weights)

        total = (self.ce_weight * l_ce
                 + self.lovasz_weight * l_lovasz
                 + self.gate_entropy_weight * l_gate)

        return {
            'total': total,
            'ce': l_ce.detach(),
            'lovasz': l_lovasz.detach(),
            'gate_entropy': l_gate.detach(),
        }
