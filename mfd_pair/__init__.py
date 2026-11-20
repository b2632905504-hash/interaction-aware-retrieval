"""MFD+PAIR retrieval, field selection, and prompt construction."""
from .interaction import InteractionAwareRetriever
from .pipeline import MFDPairRetriever, RetrievalResult, parse_prediction, save_corpus
__all__ = ["InteractionAwareRetriever", "MFDPairRetriever", "RetrievalResult", "parse_prediction", "save_corpus"]
__version__ = "0.1.0"
