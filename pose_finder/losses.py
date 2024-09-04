import torch
import torch.nn as nn


class Loss_Func(nn.Module):
    """
    Custom loss function
    """

    def __init__(self):
        super(Loss_Func, self).__init__()

    def l1_norm(self, vector):
        """
        Calculates the l1-norm

        Args:
            `vector`: Vector to apply norm to

        Returns:
            Loss calculated
        """
        loss = torch.norm(vector, p=1, dim=1)

        # Get the mean loss to account for batch size
        loss = torch.mean(loss)
        return loss

    def calc_global_rotation_loss(self, predicted, target):
        """
        Loss for global hand rotation

        Args:
            `predicted`: The predicted labels
            `target`: The target labels

        Returns:
            Loss calculated
        """
        scaler = 1

        p0 = predicted[:, 0, :]
        t0 = target[:, 0, :]

        loss = scaler * self.l1_norm(p0 - t0)
        return loss

    def calc_pose_loss(self, predicted, target):
        """
        Loss for overal hand pose

        Args:
            `predicted`: The predicted labels
            `target`: The target labels

        Returns:
            Loss calculated
        """
        scaler = 1

        p0 = predicted
        t0 = target

        loss = scaler * self.l1_norm(p0 - t0)
        return loss

    def forward(self, predicted, target):
        """
        Forward function.

        Args:
            `predicted`: The predicted labels
            `target`: The target labels
        Returns:
            Loss calculated
        """

        rot_loss = self.calc_global_rotation_loss(predicted, target)
        pose_loss = self.calc_pose_loss(predicted, target)

        return pose_loss + rot_loss
