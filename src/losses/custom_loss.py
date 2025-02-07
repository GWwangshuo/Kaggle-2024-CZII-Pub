from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
import torch.nn as nn
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, look_up_option, pytorch_after
from torch.nn.modules.loss import _Loss


class TverskyLoss(_Loss):
    """
    Compute the Tversky loss defined in:

        Sadegh et al. (2017) Tversky loss function for image segmentation
        using 3D fully convolutional deep networks. (https://arxiv.org/abs/1706.05721)

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L631

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ignore_index: int | None = None,  # 新增参数，用于指定要忽略的类别索引
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            alpha: weight of false positives
            beta: weight of false negatives
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """

        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.alpha = alpha
        self.beta = beta
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.ignore_index = ignore_index  # 添加到类的成员变量中

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if self.ignore_index is not None:
            if 0 <= self.ignore_index < n_pred_ch:
                # add ingore_index
                mask = [i for i in range(n_pred_ch) if i != self.ignore_index]
                input = input[:, mask]
                target = target[:, mask]
            else:
                warnings.warn(
                    f"ignore_index {self.ignore_index} is out of range. Valid range: [0, {n_pred_ch - 1}]. Ignoring."
                )

        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has differing shape ({target.shape}) from input ({input.shape})"
            )

        p0 = input
        p1 = 1 - p0
        g0 = target
        g1 = 1 - g0

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        tp = torch.sum(p0 * g0, reduce_axis)
        fp = self.alpha * torch.sum(p0 * g1, reduce_axis)
        fn = self.beta * torch.sum(p1 * g0, reduce_axis)
        numerator = tp + self.smooth_nr
        denominator = tp + fp + fn + self.smooth_dr

        score: torch.Tensor = 1.0 - numerator / denominator

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(score)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return score  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(score)
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )


class CELoss(_Loss):

    def __init__(
        self,
        reduction: str = "mean",
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value

        if pytorch_after(1, 10):
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=weight,
                reduction=reduction,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
            )
        else:
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=weight, reduction=reduction, ignore_index=ignore_index
            )

        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        ce_loss = (
            self.ce(input, target) if input.shape[1] != 1 else self.bce(input, target)
        )

        return ce_loss


class CustomLoss(nn.Module):
    def __init__(
        self,
        alpha=0.4,
        beta=0.6,
        tversky_coef=1.0,
        ce_coef=1.0,
        ignore_index=2,
        weight=None,
    ):
        super().__init__()
        self.tversky_coef = tversky_coef
        self.ce_coef = ce_coef
        self.ignore_index = ignore_index
        self.weight = torch.tensor(weight) if weight is not None else None

        self.tversky_loss = TverskyLoss(
            alpha=alpha,
            beta=beta,
            ignore_index=ignore_index,
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )

        self.ce_loss = CELoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        loss_dict = {}
        loss = 0.0

        if self.tversky_coef > 0:
            loss_dict["tversky_loss"] = self.tversky_coef * self.tversky_loss(
                preds, targets
            )
            loss += loss_dict["tversky_loss"]

        if self.ce_coef > 0:
            loss_dict["ce_loss"] = self.ce_coef * self.ce_loss(preds, targets)
            loss += loss_dict["ce_loss"]

        loss_dict["loss"] = loss
        return loss_dict
