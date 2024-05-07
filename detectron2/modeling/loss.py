import torch
import math
from torch.nn import functional as F

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
    challenge_factor: float = 2.0,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        challenge_factor:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()

    # Calculate the original sigmoid probabilities
    sigmoid_probs = torch.sigmoid(inputs)

    # Calculate the prediction error
    prediction_error = torch.abs(sigmoid_probs - targets)

    # Find challenging samples: those with prediction error > 0.4 and < 0.6
    challenging_samples = (prediction_error > 0.4) & (prediction_error < 0.6)

    # Adjust logits for challenging samples
    adjustment_factor = torch.where(targets[challenging_samples] == 1, challenge_factor, -challenge_factor)
    inputs[challenging_samples] += adjustment_factor

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_focal_loss)


def sigmoid_focal_loss_star(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_star_jit: "torch.jit.ScriptModule" = torch.jit.script(
    sigmoid_focal_loss_star
)

def siou_loss(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        reduction: str = "none",
        iou_weight: float = 2.5,
        eps: float = 1e-7,
) -> torch.Tensor:
    """
    Siou Loss - a combination of IoU loss with additional consideration of
    box shape, size, and angle.

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero.
    """
    # Unpack coordinates
    
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection area
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iou = intsctk / (unionk + eps)

    # Convex hull
    cw = torch.max(x2, x2g) - torch.min(x1, x1g)
    ch = torch.max(y2, y2g) - torch.min(y1, y1g)

    # Additional terms for Siou
    s_cw = (x1g + x2g - x1 - x2) * 0.5
    s_ch = (y1g + y2g - y1 - y2) * 0.5
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2 + eps)
    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
    omiga_w = torch.abs((x2 - x1) - (x2g - x1g)) / torch.max(x2 - x1, x2g - x1g)
    omiga_h = torch.abs((y2 - y1) - (y2g - y1g)) / torch.max(y2 - y1, y2g - y1g)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

    # Siou Loss
    siou = (1 - (iou - 0.5 * (distance_cost + shape_cost)))* iou_weight

    # Apply reduction
    if reduction == "mean":
        siou = siou.mean() if siou.numel() > 0 else 0.0 * siou.sum()
    elif reduction == "sum":
        siou = siou.sum()

    return siou