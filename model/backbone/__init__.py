
from .match_SEA_litenew import Matchformer_SEA_lite
from .match_SEA_large import Matchformer_SEA_large


def build_backbone():

    return Matchformer_SEA_lite()
