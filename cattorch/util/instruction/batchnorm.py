"""
Batch normalization instruction (eval mode only).

Handles both BatchNorm1d and BatchNorm2d — torch exports both as
aten.batch_norm.default.  BatchNorm1d inputs (N, C, L) are treated
as (N, C, 1, L), similar to the Conv1d approach.

The template expects:
  T1 = input, T2 = weight, T3 = bias,
  T4 = running_mean, T5 = running_var (precomputed as sqrt(var+eps)),
  T6 = output.

A special list ``_batchnorm_indices`` is pre-filled here with the flat
starting position of each channel's contiguous block of spatial elements.

Constants:
  101 = H * W  (spatial elements per channel per batch sample)
  102 = C      (number of channels)
  103 = 1      (stride — elements are contiguous within a channel)
"""

import json

import torch

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import TemplateInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class BatchNormInstruction(TemplateInstruction):
    aten_op = "aten.batch_norm.default"
    template_name = "batchnorm"

    def prepare(self):
        input_shape = self.args[0].shape  # (N, C, ...) 3-D or 4-D
        self.N = input_shape[0]
        self.C = input_shape[1]
        if len(input_shape) == 4:
            self.H = input_shape[2]
            self.W = input_shape[3]
        else:
            # BatchNorm1d: (N, C, L)
            self.H = 1
            self.W = input_shape[2]

    def transform_weights(self, static_lists):
        # Precompute sqrt(running_var + eps) so the Scratch template only
        # needs a division instead of computing sqrt at runtime.
        eps = self.args[7].value  # eps scalar
        var_key = self.args[4].name  # e.g. "W_b_bn_running_var"
        # Strip the W_ prefix that the transpiler adds
        raw_key = var_key[2:] if var_key.startswith("W_") else var_key
        if raw_key in static_lists:
            static_lists[raw_key] = torch.sqrt(static_lists[raw_key] + eps)

    def get_constants(self):
        return {
            101: self.H * self.W,
            102: self.C,
            103: 1,
        }

    def finalize(self):
        template_path = TEMPLATE_DIR / self.template_name / "template.json"
        with open(template_path) as f:
            data = json.load(f)
        data = ConstantReplacer(self.get_constants()).apply(data)

        # Pre-fill _batchnorm_indices with the flat starting position of
        # each channel's contiguous spatial block.
        # For (N, C, H, W) row-major: channel c starts at c * H * W
        # (within each batch sample, channels are laid out contiguously).
        hw = self.H * self.W
        indices = [c * hw for c in range(self.C)]

        for list_id, entry in data["lists"].items():
            if entry[0] == "_batchnorm_indices":
                entry[1] = indices
                break

        return data
