"""
AlignTac: Memory-Augmented Tactile Representation Learning

This module implements the AlignTac architecture for learning disentangled
geometric and dynamic tactile representations with memory augmentation.
"""

from .memory_module import MemoryModule
from .mdmap_decoder import MDMapDecoder
from .memory_vqvae import MemoryVQVAE

__all__ = ['MemoryModule', 'MDMapDecoder', 'MemoryVQVAE']
