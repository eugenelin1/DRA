# DR-A (Dimensionality Reduction with Adversarial variational autoencoder)

Code for the paper entitled “A deep adversarial variational autoencoder model for dimensionality reduction in single-cell RNA sequencing analysis” by Eugene Lin, Sudipto Mukherjee, and Sreeram Kannan. 

For example, a model can be trained as follows:

python dra.py --model dra --batch_size 128 --learning_rate 0.0008 --beta1 0.9 --n_l 4 --g_h_l1 512 --g_h_l2 512 --g_h_l3 512 --g_h_l4 512 --d_h_l1 32 --d_h_l2 32 --d_h_l3 32 --d_h_l4 32 --bn False --actv sig --trans sparse --keep 0.9 --leak 0.2 --lam 1.0 --epoch 200 --z_dim 2 --train --dataset Zeisel
