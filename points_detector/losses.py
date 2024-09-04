import torch
import torch.nn as nn

import bmc_loss


class Loss_Func(nn.Module):
    def __init__(self, loss_type, loss_weights):
        """
        Initializes Loss_Func.

        Args:
            `loss_type`: Type of loss to use
            `loss_weights`: Weighting to apply on losses
        """
        super(Loss_Func, self).__init__()
        self.loss_type = loss_type
        self.loss_weights = loss_weights

        invalid_weights = False
        if loss_type:
            if loss_type not in ["a", "t", "f"]:
                raise ValueError("Invalid loss type")

            if loss_type in ["a", "t"] and len(loss_weights) < 2:
                invalid_weights = True
            elif len(loss_weights) < 1:
                invalid_weights = True
        else:
            if len(loss_weights) < 1:
                invalid_weights = True

        if invalid_weights:
            raise ValueError("Not enough weights for loss_type")

    def focal_loss(self, predicted, target):
        """
        Focal loss.

        Args:
            `predicted`: The predicted labels
            `target`: The target labels

        Returns:
            Loss calculated
        """
        ce_loss = nn.L1Loss(reduction="none")(predicted, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        return torch.mean(focal_loss)

    def l1_loss(self, predicted, target):
        """
        L1-loss.

        Args:
            `predicted`: The predicted labels
            `target`: The target labels

        Returns:
            Loss calculated
        """
        l1_norm = torch.norm(predicted - target, p=1, dim=1)

        return torch.mean(l1_norm)

    def forward(self, predicted, target):
        """
        Forward function.

        Args:
            `predicted`: The predicted labels
            `target`: The target labels
        Returns:
            Loss calculated
        """

        bmc_func = bmc_loss.BMCLoss()

        l1_loss = self.l1_loss(predicted, target)
        total_bmc, bmc_dict = bmc_func.compute_loss(predicted)

        if not self.loss_type:
            return self.loss_weights[0] * l1_loss
        elif self.loss_type == "a":
            return (
                self.loss_weights[0] * l1_loss
                + self.loss_weights[1] * bmc_dict["bmc_a"]
            )

        elif self.loss_type == "t":
            return self.loss_weights[0] * l1_loss + self.loss_weights[1] * total_bmc
        elif self.loss_type == "f":
            focal_loss = self.focal_loss(predicted, target)
            return self.loss_weights[0] * focal_loss
