#!/bin/bash
LRDUAL=0.0005
EPOCHS=1000
DEVICE="cuda:1"
for huber_a in 2.0 0.5
do
    for seed in 3
    do  
        EPS=0.2
        for model in "mlp"
            do
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset translated_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset fmnist_r180 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset fmnist_r90 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset scaled_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            done
        EPS=0.1
        for model in "cnn"
            do
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset translated_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset fmnist_r180 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset fmnist_r90 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset scaled_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --penalization quad --huber_a $huber_a --method resilient --dataset fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
            done
    done
done