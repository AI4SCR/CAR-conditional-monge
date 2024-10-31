from cmonge.trainers.conditional_monge_trainer import (
    ConditionalMongeTrainer as _ConditionalMongeTrainer,
)
from cmonge.models.embedding import BaseEmbedding
from carot.models.embedding import EmbeddingFactory
from typing import Dict


class ConditionalMongeTrainer(_ConditionalMongeTrainer):
    embedding_factory: Dict[str, BaseEmbedding] = EmbeddingFactory
