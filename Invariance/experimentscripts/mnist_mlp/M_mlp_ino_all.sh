#!/bin/bash
N_AUG=2
for SEED in 1 2 3
do
python classification_image.py --method augerino --dataset translated_mnist --n_epochs 1000 --device cuda --n_samples_aug $N_AUG --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --project ALL_Inv_Augerino
python classification_image.py --method augerino --dataset mnist_r180 --n_epochs 1000 --device cuda --n_samples_aug $N_AUG --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --project ALL_Inv_Augerino
python classification_image.py --method augerino --dataset mnist_r90 --n_epochs 1000 --device cuda --n_samples_aug $N_AUG --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --project ALL_Inv_Augerino
python classification_image.py --method augerino --dataset scaled_mnist --n_epochs 1000 --device cuda --n_samples_aug $N_AUG --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --project ALL_Inv_Augerino
python classification_image.py --method augerino --dataset mnist --n_epochs 1000 --device cuda --n_samples_aug $N_AUG --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --project ALL_Inv_Augerino
done