"""Compatibility import for the canonical RandLA-Net implementation.

The former duplicate had diverged from :mod:`RandLANet_CB` and repeated each
center feature instead of gathering neighbor features. Keeping one canonical
implementation prevents training and inference from silently selecting
different model semantics.
"""

try:
    from .RandLANet_CB import RandLANet, SharedMLP, input_norm
except ImportError:
    from RandLANet_CB import RandLANet, SharedMLP, input_norm

__all__ = ["RandLANet", "SharedMLP", "input_norm"]
