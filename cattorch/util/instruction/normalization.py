"""
Normalization instructions: LayerNorm, RMSNorm, BatchNorm.

All three share a group-normalize pattern — iterate over groups of
contiguous elements and apply per-group (or per-channel) statistics.
"""

import math

import torch

from cattorch.util.instruction.instruction import TemplateInstruction


class LayerNormInstruction(TemplateInstruction):
    aten_op = "aten.layer_norm.default"
    template_name = "layernorm"

    def prepare(self):
        normalized_shape = self.args[1].value
        self.norm_size = math.prod(normalized_shape)
        self.num_groups = math.prod(self.args[0].shape) // self.norm_size

    def get_constants(self):
        return {101: self.norm_size, 102: self.num_groups}


class RMSNormInstruction(TemplateInstruction):
    aten_op = "aten.rms_norm.default"
    template_name = "rms_norm"

    def prepare(self):
        normalized_shape = self.args[1].value
        self.norm_size = math.prod(normalized_shape)
        self.num_groups = math.prod(self.args[0].shape) // self.norm_size

    def get_constants(self):
        return {101: self.norm_size, 102: self.num_groups}


class BatchNormInstruction(TemplateInstruction):
    """Batch normalization (eval mode only).

    Handles both BatchNorm1d and BatchNorm2d — torch exports both as
    aten.batch_norm.default.  BatchNorm1d inputs (N, C, L) are treated
    as (N, C, 1, L).

    The template walks sequentially through channels, just like LayerNorm
    walks through normalization groups.  Constants:
      101 = H * W  (spatial elements per channel)
      102 = C      (number of channels)
    """
    aten_op = "aten.batch_norm.default"
    template_name = "batchnorm"

    def prepare(self):
        input_shape = self.args[0].shape
        self.C = input_shape[1]
        if len(input_shape) == 4:
            self.H = input_shape[2]
            self.W = input_shape[3]
        else:
            self.H = 1
            self.W = input_shape[2]

    def transform_weights(self, static_lists):
        eps = self.args[7].value
        var_key = self.args[4].name
        raw_key = var_key[2:] if var_key.startswith("W_") else var_key
        if raw_key in static_lists:
            static_lists[raw_key] = torch.sqrt(static_lists[raw_key] + eps)

    def get_constants(self):
        return {
            101: self.H * self.W,
            102: self.C,
        }
