#!/bin/bash
EPS=0.1
LRDUAL=0.0005
EPOCHS=1000
for seed in 1 2 3
do
    for model in "cnn"
        do
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset mnist_r90 --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset translated_mnist --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset mnist_r180 --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset scaled_mnist --n_epochs $EPOCHS --device cuda --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        done
    done
done