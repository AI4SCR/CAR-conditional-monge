import os
from pathlib import Path

from cmonge.trainers.ot_trainer import MongeMapTrainer
from cmonge.utils import load_config

from carot.datasets.conditional_loader import ConditionalDataModule
from carot.datasets.single_loader import CarModule
from carot.trainers.conditional_monge_trainer import ConditionalMongeTrainer


def test_conditional_model_training(cond_synthetic_config):

    config = load_config(cond_synthetic_config)

    datamodule = ConditionalDataModule(config.data, config.condition)

    logger_path = Path(config.logger_path)

    datamodule = ConditionalDataModule(config.data, config.condition)
    trainer = ConditionalMongeTrainer(
        jobid=1, logger_path=logger_path, config=config.model, datamodule=datamodule
    )

    trainer.train(datamodule)
    trainer.evaluate(datamodule)

    os.remove(logger_path)


def test_model_training(synthetic_config):

    config = load_config(synthetic_config)

    datamodule = CarModule(config.data)

    logger_path = Path(config.logger_path)

    datamodule = MongeMapTrainer(config.data, config.condition)
    trainer = MongeMapTrainer(jobid=1, logger_path=logger_path, config=config.model)

    trainer.train(datamodule)
    trainer.evaluate(datamodule)

    os.remove(logger_path)
