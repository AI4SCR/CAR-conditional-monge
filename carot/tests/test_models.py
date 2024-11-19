import os
from pathlib import Path

from cmonge.trainers.ot_trainer import MongeMapTrainer

from carot.datasets.conditional_loader import ConditionalDataModule
from carot.datasets.single_loader import CarModule
from carot.trainers.conditional_monge_trainer import ConditionalMongeTrainer


def test_conditional_model_training(cond_synthetic_config):

    datamodule = ConditionalDataModule(
        cond_synthetic_config.data, cond_synthetic_config.condition
    )

    logger_path = Path(cond_synthetic_config.logger_path)

    datamodule = ConditionalDataModule(
        cond_synthetic_config.data, cond_synthetic_config.condition
    )
    trainer = ConditionalMongeTrainer(
        jobid=1,
        logger_path=logger_path,
        config=cond_synthetic_config.model,
        datamodule=datamodule,
    )

    trainer.train(datamodule)
    trainer.evaluate(datamodule)

    os.remove(logger_path)


def test_model_training(synthetic_config):

    datamodule = CarModule(synthetic_config.data)

    logger_path = Path(synthetic_config.logger_path)

    datamodule = MongeMapTrainer(synthetic_config.data, synthetic_config.condition)
    trainer = MongeMapTrainer(
        jobid=1, logger_path=logger_path, config=synthetic_config.model
    )

    trainer.train(datamodule)
    trainer.evaluate(datamodule)

    os.remove(logger_path)
