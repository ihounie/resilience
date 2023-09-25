#!/bin/bash
EPS=0.1
LRDUAL=0.0005
EPOCHS=1000
for huber_a in 0.5
do
    for seed in 0 1 2:
    do
        for model in "mlp" "cnn"
        do
            python classification_image.py --penalization quad --huber_a $huber_a --lr_dual $LRDUAL --epsilon $EPS --model $model --method resilient --dataset mnist_r90 --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --penalization quad --huber_a $huber_a --lr_dual $LRDUAL --epsilon $EPS --model $model --method resilient --dataset translated_mnist --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --penalization quad --huber_a $huber_a --lr_dual $LRDUAL --epsilon $EPS --model $model --method resilient --dataset mnist_r180 --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --penalization quad --huber_a $huber_a --lr_dual $LRDUAL --epsilon $EPS --model $model --method resilient --dataset scaled_mnist --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        done
    done
done