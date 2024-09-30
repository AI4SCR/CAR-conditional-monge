from pathlib import Path

from cmonge.datasets.single_loader import CarModule
from cmonge.trainers.ot_trainer import MongeMapTrainer

from cmonge.utils import load_config

from orbax.checkpoint.test_utils import erase_and_create_empty
from loguru import logger

import typer


def train_conditional_monge(config_path: Path):

    config = load_config(config_path)

    # For submitting jobs
    if len(config.logger_path) != 0:
        logger_path = Path(config.logger_path)
    else:
        data = config.data.file_path.split("/")[-1][:-5]
        logger_path = Path(
            "/cmongelogs/monge/{data}_{config.data.drug_condition}_monge.yml"
        )
    logger.info(f"Experiment: Training model on {config.data.drug_condition}")

    # Train initial model
    erase_and_create_empty(config.model.checkpointing_path)
    datamodule = CarModule(config.data)
    trainer = MongeMapTrainer(jobid=1, logger_path=logger_path, config=config.model)
    trainer.train(datamodule)
    trainer.save_checkpoint(config=config.model)

    # Reload from best checkpoint
    # if config.model.checkpointing:
    trainer = MongeMapTrainer.load_checkpoint(
        jobid=1, config=config.model, logger_path=logger_path
    )
    trainer.evaluate(datamodule, valid=True)

    # Evaluate baselines
    # Model doesn't need to be trained, as there is no transport happening
    trainer.evaluate(datamodule, identity=True, valid=True)
    config.data.control_condition = config.data.drug_condition
    datamodule = CarModule(config.data)
    trainer.evaluate(datamodule, identity=True, valid=True)

    print("Training completed")


if __name__ == "__main__":
    typer.run(train_conditional_monge)
