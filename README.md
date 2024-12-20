# Chimeric Antigen Receptor Optimal Transport (CAROT)

[![CI](https://github.com/AI4SCR/car-conditional-monge/actions/workflows/ci.yml/badge.svg)](https://github.com/AI4SCR/car-conditional-monge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The conditional Monge Gap applied the single cell RNA sequencing data of Chimeric Antigen Receptor T cells. Extension of the [Conditional Monge Gap](https://github.com/AI4SCR/conditional-monge), to include CAR specific dataloaders, embeddings and trainers. Additionally `notebooks` contains Notebooks for generating the figures of the preprint ... and additional analyses. In the `configs` and `scripts` directories are all scripts to replicate the experiments from this preprint.

## Development setup & installation
We use [poetry](https://python-poetry.org/docs/managing-environments/) as package manager and tested the code in Python 3.10.
```sh
pip install poetry # into your base env
git clone git@github.com:AI4SCR/car-conditional-monge.git
cd car-conditional-monge
poetry install -v
```

If the installation was successful, activate the env interatively via `poetry shell`.

## TODO: Add usage example

## Citation
If you find this work useful, please cite:


```bib
@inproceedings{driessen2024modeling,
  title={Modeling CAR Response at the Single-Cell Level Using Conditional OT},
  author={Driessen, Alice and Born, Jannis and Rueda, Roc{\'\i}o Castellanos and Reddy, Sai T and Rapsomaniki, Marianna},
  year={2024},
  booktitle={NeurIPS 2024 Workshop on AI for New Drug Modalities},
  note={Spotlight talk}
}
```