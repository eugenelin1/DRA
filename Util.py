import tables
import scipy.sparse
import scipy.io
from sklearn.decomposition import PCA
import numpy as np
import pdb
import os
from scipy import * 


def next_batch(data, batch_size, max_size):
    indx = np.random.randint(max_size-batch_size)
    return data[indx:(indx+batch_size), :]

def next_batch_labelled(data, label, batch_size, max_size):
    indx = np.random.randint(max_size - batch_size)
    return data[indx:(indx+batch_size), :], label[indx:(indx+batch_size)]

def read_mtx(filename, trans_flag = False):
     buffer = scipy.io.mmread(filename)

     if trans_flag:
         print('Transpose Data !')
         return buffer.transpose()
     else:
         return buffer


def write_mtx(filename, data):
    scipy.io.mmwrite(filename, data)


def load_gene_mtx(dataset_name, transform = True, count = True, actv = 'sig'):
    data_path = './data/'+ dataset_name +'/sub_set-720.mtx'
    data = read_mtx(data_path)
    print('Data Loaded from {} !'.format(data_path))
    data = data.toarray()

    if transform:
        data = transform_01_gene_input(data)
        print('Data Transformed, entries in [0, 1] !'.format(data_path))
    else:
        if count == False:
            data = np.log2(data+1)

            if actv == 'lin':
                scale = 1.0
            else:
                scale = np.max(data)
            data = data / scale           

    total_size = data.shape[0]
    if dataset_name == '10x_73k': 
        train_size = 58586 
        val_size = 0 

    elif dataset_name == '10x_68k': 
        train_size = 54863 
        val_size = 0 
        
    elif dataset_name == 'Macosko': 
        train_size = 35846 
        val_size = 0 

    elif dataset_name == 'Zeisel': 
        train_size = 2404 
        val_size = 0 

    np.random.seed(0)
    indx = np.random.permutation(np.arange(total_size))
    data_train = data[indx[0:train_size], :]
    data_val = data[indx[train_size:train_size + val_size], :]
    data_test = data[indx[train_size + val_size:], :]

    if count == False:
        return data_train, data_val, data_test, scale

    return data_train, data_val, data_test


def load_labels(dataset_name):
    data_path = './data/' + dataset_name + '/labels.txt'
  
    labels = np.loadtxt(data_path)

    total_size = labels.shape[0]

    if dataset_name == '10x_73k': 
        train_size = 58586 
        val_size = 0 

    elif dataset_name == '10x_68k': 
        train_size = 54863 
        val_size = 0 
        
    elif dataset_name == 'Macosko': 
        train_size = 35846 
        val_size = 0 

    elif dataset_name == 'Zeisel': 
        train_size = 2404 
        val_size = 0 

    np.random.seed(0)
    indx = np.random.permutation(np.arange(total_size))
    labels_train = labels[indx[0:train_size]]
    labels_val = labels[indx[train_size:train_size + val_size]]
    labels_test = labels[indx[train_size + val_size:]]
    return labels_train, labels_val, labels_test













