# DR-A (Dimensionality Reduction with Adversarial variational autoencoder)

Code for the paper entitled “A deep adversarial variational autoencoder model for dimensionality reduction in single-cell RNA sequencing analysis” by Eugene Lin, Sudipto Mukherjee, and Sreeram Kannan. 


## Dependencies

The code has been tested with the following versions of packages.

- Python 2.7.12
- Tensorflow 1.4.0
- Numpy 1.14.2


## Code for DR-A

There are three python files for DR-A as follows:

- dra.py
- Util.py
- opts.py

The path for the code could be ./DRA/<python_file_name.py>
We utiize a fixed seed using tf.random.seed(0).


## Dataset

The path for the dataset could be ./DRA/data/<dataset_name>
For example, the Zeisel dataset could be in the folder ./DRA/data/Zeisel as follows:

- ./DRA/data/Zeisel/sub_set-720.mtx
- ./DRA/data/Zeisel/labels.txt

The Zeisel dataset consists of 3,005 cells from the mouse brain (as shown in the sub_set-720.mtx file). 
In addition, the Zeisel dataset has the ground truth labels of 7 distinct cell types (as shown in the labels.txt file).
There were 720 highest variance genes selected in the Zeisel dataset.


## Training

For example, a model can be trained using the command as follows:

```bash
$ python dra.py --model dra --batch_size 128 --learning_rate 0.0008 --beta1 0.9 --n_l 4 --g_h_l1 512 --g_h_l2 512 --g_h_l3 512 --g_h_l4 512 --d_h_l1 32 --d_h_l2 32 --d_h_l3 32 --d_h_l4 32 --bn False --actv sig --trans sparse --keep 0.9 --leak 0.2 --lam 1.0 --epoch 200 --z_dim 2 --train --dataset Zeisel
```
The above model is trained using the hyperparameters as follows:

- Latent dimensions = 2 (using the option --z_dim)
- Batch size = 128 (using the option --batch_size)
- Learning rate = 0.0008 (using the option --learning_rate)
- Hidden layer = 4 (using the option --n_l)
- Hidden unit for the generator = 512/512/512/512 (using the option --g_h_l1, --g_h_l2, --g_h_l3, and --g_h_l4)
- Hidden unit for the discriminator = 32/32/32/32 (using the option --d_h_l1, --d_h_l2, --d_h_l3, and --d_h_l4)

The performance will be written to ./DRA/Res_DRA/tune_logs/Metrics_Zeisel.txt along with timestamp.
We assess the clustering performance using the normalized mutual information (NMI) scores (as shown in the Metrics_Zeisel.txt file).


Furthermore, an example for another model can be trained using the command as follows:

```bash
$ python dra.py --model dra --batch_size 128 --learning_rate 0.0007 --beta1 0.9 --n_l 1 --g_h_l1 512 --d_h_l1 512 --bn False --actv sig --trans sparse --keep 0.9 --leak 0.2 --lam 1.0 --epoch 200 --z_dim 10 --train --dataset Zeisel
```
The second model is trained using the hyperparameters as follows:

- Latent dimensions = 10 (using the option --z_dim)
- Batch size = 128 (using the option --batch_size)
- Learning rate = 0.0007 (using the option --learning_rate)
- Hidden layer = 1 (using the option --n_l)
- Hidden unit for the generator = 512 (using the option --g_h_l1)
- Hidden unit for the discriminator = 512 (using the option --d_h_l1)

Again, the performance will be written to ./DRA/Res_DRA/tune_logs/Metrics_Zeisel.txt along with timestamp.
We assess the clustering performance using the normalized mutual information (NMI) scores (as shown in the Metrics_Zeisel.txt file).

Moreover, there are several other options as follows:

- Epoch to train (using the option --epoch)
- Momentum term of the ADAM algorithm (using the option --beta1)
- Activation function for the decoder (using --actv)
- Leak factor (using --leak)
- Keep probability (using --keep)
- Data transformation (using --trans)
- Batch normalization (using --bn)
- Lambda for regularization (using --lam)
- Dataset name (using --dataset)
- Generator hidden units in layer 1 (using --g_h_l1)
- Generator hidden units in layer 2 (using --g_h_l2)
- Generator hidden units in layer 3 (using --g_h_l3)
- Generator hidden units in layer 4 (using --g_h_l4)
- Discriminator hidden units in layer 1 (using --d_h_l1)
- Discriminator hidden units in layer 2 (using --d_h_l2)
- Discriminator hidden units in layer 3 (using --d_h_l3)
- Discriminator hidden units in layer 4 (using --d_h_l4)
