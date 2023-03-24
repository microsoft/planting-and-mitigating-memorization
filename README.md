# Planting and Mitigating Memorization

This repository contains the software used to run the experiments in the paper "Planting and Mitigating Memorized Content in Predictive-Text Language Models" ([pre-print](https://arxiv.org/pdf/2212.08619.pdf)). This research evaluates the propensity for user data memorization in language models under a variety of modeling and adversarial conditions, tests the efficacy of various privacy mitigations intended to reduce memorization.

## Repository Structure

- `Configs` contains yaml configurations for each experiment.
- `Canaries` contains the artificial training examples used as a test suite in this study.
- `RunScripts` contains the shell scripts used to conduct all experiments.
- `src` contains the Python scripts and classes used to train and evaluate language models
- `Tools` contains scripts to download and pre-process the text data, anonymize text with EUII-scrubbing, and evaluate model predictions
- `experiment_results.csv`: contains results published in the paper.
- `requirements.txt` in the main directory can be used to initialize the correct Python environment (Python 3.9).

## How to Cite

```bibtex
@article{downey-etal-2022,
  author        = {C.M. Downey and Wei Dai and Huseyin A. Inan and Kim Laine and Saurabh Naik and Tomasz Religa},
  title         = {Planting and Mitigating Memorized Content in Predictive-Text Language Models},
  year          = {2022},
  month         = {December},
  journal       = {arXiv:2212.08619 [cs]},
  url           = {\url{https://arxiv.org/abs/2212.08619}}
}
```

## Contributing

For contributing to this repository, see [CONTRIBUTING](CONTRIBUTING.md).
