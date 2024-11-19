import os
from pathlib import Path

import pytest
from cmonge.trainers.ot_trainer import MongeMapTrainer
from cmonge.utils import load_config

from carot.datasets.conditional_loader import ConditionalDataModule
from carot.datasets.single_loader import CarModule
from carot.trainers.conditional_monge_trainer import ConditionalMongeTrainer


@pytest.fixture(scope="session")
def synthetic_config():
    config_path = Path("carot/tests/configs/synthetic.yml")
    config = load_config(config_path)
    return config


@pytest.fixture(scope="session")
def cond_synthetic_config():
    config_path = Path("carot/tests/configs/conditional_synthetic.yml")
    config = load_config(config_path)
    return config


@pytest.fixture(scope="module")
def synthetic_data(synthetic_config):
    module = CarModule(synthetic_config.data)
    return module


@pytest.fixture(scope="module")
def cond_synthetic_data(cond_synthetic_config):
    module = ConditionalDataModule(cond_synthetic_config.data)
    return module


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

    datamodule = ConditionalDataModule(config.data, config.condition)

    logger_path = Path(config.logger_path)

    datamodule = MongeMapTrainer(config.data, config.condition)
    trainer = MongeMapTrainer(jobid=1, logger_path=logger_path, config=config.model)

    trainer.train(datamodule)
    trainer.evaluate(datamodule)

    os.remove(logger_path)
