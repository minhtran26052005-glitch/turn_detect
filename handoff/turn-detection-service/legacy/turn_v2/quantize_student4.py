"""Weight-only INT4 quantization used by QAT."""
import ml_dtypes
import numpy as np

def quantize_groups(values, group_size, scale_method="mse"):
    """Return signed INT4 codes and FP16 scales for last-axis groups."""
    width = values.shape[-1]
    groups = (width + group_size - 1) // group_size
    padded = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, groups * group_size - width)])
    blocks = padded.reshape(*values.shape[:-1], groups, group_size)
    valid = np.arange(groups * group_size).reshape(groups, group_size) < width
    valid = valid.reshape((1,) * (values.ndim - 1) + valid.shape)
    scale = np.maximum(np.max(blocks, axis=-1) / 7, -np.min(blocks, axis=-1) / 8)
    scale = np.maximum(scale, np.finfo(np.float16).tiny)
    best_scale = scale.astype(np.float16)
    best_error = np.full(scale.shape, np.inf, np.float32)
    # Fixed weight-only search, with rounded deployment scales in the objective.
    for fraction in ([1.] if scale_method == "max" else [.8, .85, .9, .95, 1.]):
        candidate = np.maximum(scale * fraction, np.finfo(np.float16).tiny).astype(np.float16)
        codes = np.clip(np.rint(blocks / candidate[..., None].astype(np.float32)), -8, 7)
        error = (((codes * candidate[..., None].astype(np.float32) - blocks) ** 2) * valid).sum(-1)
        better = error < best_error
        best_error = np.minimum(best_error, error)
        best_scale = np.where(better, candidate, best_scale)
    codes = np.clip(np.rint(blocks / best_scale[..., None].astype(np.float32)), -8, 7)
    codes = codes.reshape(*values.shape[:-1], groups * group_size)[..., :width]
    restored_scale = np.repeat(best_scale.astype(np.float32), group_size, axis=-1)[..., :width]
    restored = codes * restored_scale
    return codes.astype(ml_dtypes.int4), best_scale, restored
