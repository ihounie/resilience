#!/bin/bash
for seed in 1 2 3
do
    for model in "mlp" "cnn"
        do
        python classification_image.py --model $model --method uniform_aug --dataset translated_fmnist  --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        python classification_image.py --model $model --method uniform_aug --dataset fmnist_r180 --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        python classification_image.py --model $model --method uniform_aug --dataset fmnist_r90 --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        python classification_image.py --model $model --method uniform_aug --dataset scaled_fmnist  --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        for sz in 312 1250 5000 20000
            do
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset translated_fmnist  --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset fmnist_r180 --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset fmnist_r90 --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset scaled_fmnist  --n_epochs 1000 --device cuda:1 --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            done
        done
    done
done