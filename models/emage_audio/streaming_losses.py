import torch


def velocity_loss(pred, target):
    """Match first-order motion dynamics."""
    pred_v = pred[:, 1:] - pred[:, :-1]
    target_v = target[:, 1:] - target[:, :-1]
    return torch.nn.functional.l1_loss(pred_v, target_v)


def acceleration_loss(pred, target):
    """Match second-order gesture dynamics."""
    pred_a = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    target_a = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    return torch.nn.functional.l1_loss(pred_a, target_a)


def boundary_consistency_loss(chunk_a, chunk_b):
    """Reduce discontinuity between consecutive streaming chunks."""
    return torch.nn.functional.l1_loss(chunk_a[:, -1], chunk_b[:, 0])


def streaming_motion_loss(pred, target, velocity_weight=0.2, accel_weight=0.05):
    return (
        torch.nn.functional.l1_loss(pred, target)
        + velocity_weight * velocity_loss(pred, target)
        + accel_weight * acceleration_loss(pred, target)
    )
