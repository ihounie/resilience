# Resilient Constrained Learning

This repository contains the code for the Invariance Constrained Learning Experiments (described on Appendix H).

It is based on the [official repo](https://github.com/tychovdo/lila.git) for the paper
"[Invariance Learning in Deep Neural Networks with Differentiable Laplace Approximations](https://arxiv.org/abs/2202.10638)" by T. van der Ouderaa*, A. Immer*, M. Welling, and M. Filippone.

## Setup

Python 3.8 and pip is required.

```bash

pip install  -r  requirements.txt

```
Create directory for results: `mkdir results` in the root of the project.

We have added weights and biases logging to the code. To use this, you need to create a free account at [wandb.ai](https://wandb.ai/). Then, you need to login using `wandb login` and follow the instructions.

## Running experiments

Bash scripts for running the experiments in the paper are in the `experimentscripts` directory. The scripts are organized by dataset and method.

All experiments run clasificatio_image.py with different arguments.

We have added/changes the following arguments to the original code to accomodate our method:

- `--method` : method to use. Options are `constrained`, `resilient`, `augerino`, `baseline`

- `--epsilon` : epsilon for the constrained formulation

- `--lr_dual` : learning rate for the dual variables

- `--lr_perturb` : learning rate for the dual variables

- `--penalization` : type of penalization for the constrained formulation. Options are `quad` and `huber` (quad is used in the paper)

- `--project` : name of the project in weights and biases.

- `--huber_alpha` : alpha for the huber and quadratic penalization

  

### MLP on rotated (90 deg) MNIST

  

To run Resilient Formulation:

```

python classification_image.py --lr_dual 0.0005 --epsilon 0.25 --model $model --method constrained --dataset mnist_r90 --n_epochs 1000 --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed 0 --penalization quad

```

  

To run Constrained Formulation:

```

python classification_image.py --lr_dual 0.0005 --epsilon 0.25 --model $model --method constrained --dataset mnist_r90 --n_epochs 1000 --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed 0

```

  
  

To run Augerino:

```

python classification_image.py --method augerino --dataset mnist_r90 --n_epochs 1000 --device cuda --n_samples_aug 2 --save --optimize_aug --approx ggn_kron --batch_size 1000 --project ALL_Inv_Augerino --seed 0

```

  

To run unconstrained baseline:

```

python classification_image.py --method baseline --dataset translated_mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --approx ggn_kron --batch_size 1000 --seed 1 --download

```