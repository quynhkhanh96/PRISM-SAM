import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    logits:  (B, H, W)
    targets: (B, H, W) in {0,1}
    """
    probs = torch.sigmoid(logits)

    num = 2 * (probs * targets).sum(dim=(1, 2))
    den = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + eps

    dice = num / den
    return 1.0 - dice  # (B,)


def mask_iou_from_logits(logits, targets, eps=1e-6):
    """
    logits:  (B, H, W)
    targets: (B, H, W)
    """
    preds = (logits > 0).float()

    inter = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - inter

    return inter / (union + eps)  # (B,)


def sigmoid_focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
    reduction="mean"
):
    """
    logits:  (B, H, W)
    targets: (B, H, W)
    """
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)

    loss = ce * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class SAMMultimaskLoss(nn.Module):
    """
    SAM-style multimask loss.

    - Best-of-K mask supervision
    - Focal + Dice on best mask
    - Optional IoU score regression
    """

    def __init__(
        self,
        focal_weight=20.0,
        dice_weight=1.0,
        score_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.score_weight = score_weight

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        mask_logits,   # (B*K, num_masks, H, W)   typically H=W=256
        gt_masks,      # (B*K, H, W)
        iou_preds=None # (B*K, K) or None
    ):
        # BK = B*K, B = number of instances in a batch, K = number of prompts per instance
        # num_masks = number of output masks, = 4 if multimask_output is True, 1 otherwise
        BK, num_masks, H, W = mask_logits.shape 
        gt_masks = gt_masks.float()

        # ---- compute IoU for each mask ----
        ious = []
        for i in range(num_masks):
            iou_i = mask_iou_from_logits(
                mask_logits[:, i], gt_masks
            )
            ious.append(iou_i)

        ious = torch.stack(ious, dim=1)  # (BK, num_masks)

        # ---- select best mask ----
        best_i = ious.argmax(dim=1)  # (BK,)
        best_logits = mask_logits[
            torch.arange(BK, device=mask_logits.device), best_i
        ]  # (BK, H, W)

        # ---- segmentation loss ----
        focal = sigmoid_focal_loss(
            best_logits,
            gt_masks,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean"
        )

        dice = dice_loss_from_logits(
            best_logits, gt_masks
        ).mean()

        seg_loss = (
            self.focal_weight * focal
            + self.dice_weight * dice
        )

        # ---- IoU score regression loss ----
        score_loss = torch.tensor(
            0.0, device=mask_logits.device
        )

        if iou_preds is not None:
            score_loss = F.mse_loss(iou_preds, ious)

        total_loss = seg_loss + self.score_weight * score_loss

        return {
            "loss": total_loss,
            "seg": seg_loss.detach(),
            "focal": focal.detach(),
            "dice": dice.detach(),
            "score": score_loss.detach(),
            "best_iou": ious.max(dim=1)[0].mean().detach(),
        }

# --- Usage ---
# criterion = SAMMultimaskLoss(
#     focal_weight=20.0,
#     dice_weight=1.0,
#     score_weight=1.0
# )

# loss_dict = criterion(
#     mask_logits,   # (B*K, num_masks, 256, 256)
#     gt_masks,      # (B*K, 256, 256)
#     iou_preds      # (B*K, num_masks) or None
# )

# loss = loss_dict["loss"]
# loss.backward()
