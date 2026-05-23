"""
ITK utility functions for medical image processing.

This module provides utility functions for working with ITK (Insight Toolkit)
images, including conversion utilities and canonical form normalization.
"""

from itk.image_ops import to_canonical

__all__ = ["to_canonical"]
