"""
Hypothesis-Specific Target Generation System
==========================================

This module contains all hypothesis-specific target generation logic,
separated from generic feature engineering for better organization.
"""

from .base_target_generator import BaseTargetGenerator
from .target_generators import TargetHypothesis

__all__ = ['BaseTargetGenerator', 'TargetHypothesis']