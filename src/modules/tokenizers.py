"""Contains tokenizers like GloveTokenizers and BERT Tokenizer."""

import torch
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset
from src.utils.mapper import configmapper
from transformers import AutoTokenizer


class Tokenizer:
    """Abstract Class for Tokenizers."""

    def tokenize(self):
        """Abstract Method for tokenization."""
