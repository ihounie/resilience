#!/bin/bash
EPOCHS=1000
for seed in 2 3
do
    for model in "mlp"
    do
    python classification_image.py --model $model --method baseline --dataset mnist_r90 --n_epochs $EPOCHS --device cuda --save --batch_size 1000 --seed $seed --project Inv_Unconstrained --wandb_log
    python classification_image.py --model $model --method baseline --dataset translated_mnist --n_epochs $EPOCHS --device cuda --save --batch_size 1000 --seed $seed --project Inv_Unconstrained --wandb_log
    python classification_image.py --model $model --method baseline --dataset mnist_r180 --n_epochs $EPOCHS --device cuda --save --batch_size 1000 --seed $seed --project Inv_Unconstrained --wandb_log
    python classification_image.py --model $model --method baseline --dataset scaled_mnist --n_epochs $EPOCHS --device cuda --save --batch_size 1000 --seed $seed --project Inv_Unconstrained --wandb_log
    done
done