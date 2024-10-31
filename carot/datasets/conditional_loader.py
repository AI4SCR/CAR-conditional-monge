from cmonge.datasets.conditional_loader import (
    ConditionalDataModule as _ConditionalDataModule,
)
from .single_loader import DataModuleFactory
from typing import Dict


class ConditionalDataModule(_ConditionalDataModule):
    datamodule_factory: Dict[str, _ConditionalDataModule] = DataModuleFactory
