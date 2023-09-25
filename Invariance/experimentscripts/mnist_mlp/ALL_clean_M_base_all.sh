#!/bin/bash
EPS=0.1
LRDUAL=0.0005
EPOCHS=1000
huber_a=1.0
for seed in 1 2 3
do
    for model in "mlp" "cnn"
        do
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset mnist --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed --project ALL_Invariance
        python classification_image.py --model $model --method baseline --dataset mnist --n_epochs $EPOCHS --device cuda --save --batch_size 1000 --seed $seed --project Inv_Unconstrained
        python classification_image.py --penalization quad --huber_a $huber_a --lr_dual $LRDUAL --epsilon $EPS --model $model --method resilient --dataset mnist --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        done
    done
done