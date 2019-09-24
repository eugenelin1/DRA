import tensorflow as tf
import os
from opts import *
import Util
import numpy as np
import scipy.sparse
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats 
from scipy import * 
import datetime 

np.random.seed(0)
tf.set_random_seed(0)


class Test_DRA(object):
    def __init__(self, sess, epoch = 200, lr=0.0001, beta1=0.5, batch_size=128, X_dim=720, z_dim=10, dataset_name='mnist',
                 checkpoint_dir='checkpoint', sample_dir='samples', result_dir = 'result', num_layers = 2, g_h_dim=None,
                 d_h_dim=None, gen_activation='sig', leak = 0.2, keep_param = 1.0, trans = 'sparse',is_bn=False,
                 g_iter = 2, lam=10.0, sampler='uniform'):

        self.sess = sess
        self.epoch = epoch
        self.lr = lr
        self.beta1 = beta1
        self.batch_size = batch_size
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.result_dir = result_dir
        self.num_layers = num_layers
        self.g_h_dim = g_h_dim  # Fully connected layers for Generator
        self.d_h_dim = d_h_dim  # Fully connected layers for Discriminator
        self.gen_activation = gen_activation
        self.leak = leak
        self.keep_param = keep_param
        self.trans = trans
        self.is_bn = is_bn
        self.g_iter = g_iter
        self.lam = lam
        self.sampler = sampler
        self.eps = 0.001
        self._is_train = False
        self.n_hidden = 128 
        

        if self.dataset_name == '10x_73k' or self.dataset_name == '10x_68k' or self.dataset_name == 'Zeisel' or self.dataset_name == 'Macosko':

                if self.trans == 'sparse':
                    self.data_train, self.data_val, self.data_test, self.scale = Util.load_gene_mtx(self.dataset_name, transform=False, count=False, actv=self.gen_activation)
                else:
                    self.data_train, self.data_val, self.data_test = Util.load_gene_mtx(self.dataset_name, transform=True)
                    self.scale = 1.0

        self.labels_train, self.labels_val, self.labels_test = Util.load_labels(self.dataset_name)
                
        if self.gen_activation == 'tanh':
            self.data = 2* self.data - 1
            self.data_train = 2 * self.data_train - 1
            self.data_val = 2 * self.data_val - 1
            self.data_test = 2 * self.data_test - 1

        self.train_size = self.data_train.shape[0]
        self.test_size = self.data_test.shape[0]
        self.total_size = self.train_size + self.test_size 

        self.data = np.concatenate([self.data_train, self.data_test])

        print("Shape self.data_train:", shape(self.data_train)) 
        print("Shape self.data_test:", shape(self.data_test)) 
    
        self.build_model()

    def build_model(self):

        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Input')
        self.x_target = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Target')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name = 'keep_prob')
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='Real_distribution')
        self.kl_scale = tf.placeholder(tf.float32, (), name='kl_scale')
        
        self.kl_scale = 0        
        self.dropout_rate = 0.1 
        self.training_phase = True 
        self.n_layers = self.num_layers 
        self.n_latent = self.z_dim
        
        self.encoder_output, self.z_post_m, self.z_post_v, self.l_post_m, self.l_post_v = self.encoder(self.x_input) 
        self.expression = self.x_input               
        self.proj = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='projection')
      
        log_library_size = np.log(np.sum(self.data_train, axis=1)) 
        mean, variance = np.mean(log_library_size), np.var(log_library_size)
        library_size_mean = mean
        library_size_variance = variance
        self.library_size_mean = tf.to_float(tf.constant(library_size_mean))
        self.library_size_variance = tf.to_float(tf.constant(library_size_variance))        
        self.z = self.sample_gaussian(self.z_post_m, self.z_post_v) 
        self.library = self.sample_gaussian(self.l_post_m, self.l_post_v)
        self.decoder_output = self.decoder(self.z)               
        self.n_input = self.expression.get_shape().as_list()[1] 
      
        self.x_post_scale = tf.nn.softmax(dense(self.decoder_output, self.g_h_dim[0], self.n_input, name='dec_x_post_scale')) 
        self.x_post_r = tf.Variable(tf.random_normal([self.n_input]), name="dec_x_post_r")           
        self.x_post_rate = tf.exp(self.library) * self.x_post_scale
        self.x_post_dropout = dense(self.decoder_output, self.g_h_dim[0], self.n_input, name='dec_x_post_dropout') 
            
        local_dispersion = tf.exp(self.x_post_r)            
        local_l_mean = self.library_size_mean
        local_l_variance = self.library_size_variance

        self.decoder_output2 = tf.nn.sigmoid(dense(self.decoder_output, self.g_h_dim[0], self.X_dim, 'dec_output2'))        
        self.dis_real_logit = self.discriminator(self.real_distribution, self.z_dim) 
        self.dis_fake_logit = self.discriminator(self.z, self.z_dim, reuse=True) 
       
        # Discriminator D2
        self.dis2_real_logit = self.discriminator2(self.x_target, self.X_dim)       
        self.dis2_fake_logit = self.discriminator2(self.decoder_output2, self.X_dim, reuse=True)         
        
        # Reconstruction loss 
        recon_loss = self.zinb_model(self.expression, self.x_post_rate, local_dispersion, self.x_post_dropout)        
        
        kl_gauss_l = 0.5 * tf.reduce_sum(- tf.log(self.l_post_v + 1e-8)  \
                                         + self.l_post_v/local_l_variance \
                                         + tf.square(self.l_post_m - local_l_mean)/local_l_variance  \
                                         + tf.log(local_l_variance + 1e-8) - 1, 1)

        kl_gauss_z = 0.5 * tf.reduce_sum(- tf.log(self.z_post_v + 1e-8) + self.z_post_v + tf.square(self.z_post_m) - 1, 1)
        
        # Evidence lower bound        
        self.ELBO_gauss = tf.reduce_mean(recon_loss - kl_gauss_l - self.kl_scale * kl_gauss_z) 
        self.autoencoder_loss = - self.ELBO_gauss                                
              
        # Discriminator D1        
        self.dis_loss = - tf.log(tf.reduce_sum(tf.sqrt(tf.abs(self.dis_real_logit/tf.reduce_sum(self.dis_real_logit)
                                            * self.dis_fake_logit/tf.reduce_sum(self.dis_fake_logit)) )) + 1e-10)        
                
        # Discriminator D2 
        self.dis2_loss = - tf.log(tf.reduce_sum(tf.sqrt(tf.abs(self.dis2_real_logit/tf.reduce_sum(self.dis2_real_logit)
                                            * self.dis2_fake_logit/tf.reduce_sum(self.dis2_fake_logit)) )) + 1e-10)
        
        # Generator loss
        self.generator_loss = - tf.log(tf.reduce_sum(tf.sqrt(tf.abs(
                                            self.dis_fake_logit/tf.reduce_sum(self.dis_fake_logit)))) + 1e-10)
                
        t_vars = tf.trainable_variables()
        self.dis_vars = [var for var in t_vars if 'dis_' in var.name]
        self.gen_vars = [var for var in t_vars if 'enc_' in var.name]

        # Discriminator D2
        self.dis2_vars = [var for var in t_vars if 'dis2_' in var.name]

        self.saver = tf.train.Saver()

    def train_cluster(self):

        print('Cluster DRA on DataSet {} ... '.format(self.dataset_name))

        autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                       beta1=self.beta1).minimize(self.autoencoder_loss)
        
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                         beta1=self.beta1).minimize(self.dis_loss,
                                                                                      var_list=self.dis_vars)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                     beta1=self.beta1).minimize(self.generator_loss,
                                                                                  var_list=self.gen_vars)
        # Discriminator D2
        discriminator2_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                         beta1=self.beta1).minimize(self.dis2_loss,
                                                                                      var_list=self.dis2_vars)

        self.sess.run(tf.global_variables_initializer())
        a_loss_epoch = []
        d_loss_epoch = []
        g_loss_epoch = []
        d2_loss_epoch = [] # Discriminator D2

        control = 3 # Generator is updated twice for each Discriminator D1 update

        num_batch_iter = self.total_size // self.batch_size
        for ep in range(self.epoch):
            d_loss_curr = g_loss_curr = a_loss_curr = np.inf
            self._is_train = True
            for it in range(num_batch_iter):

                batch_x = self.next_batch(self.data_train, self.train_size)                
                batch_z_real_dist = self.sample_Z(self.batch_size, self.z_dim)

                _, a_loss_curr = self.sess.run([autoencoder_optimizer, self.autoencoder_loss],
                                               feed_dict={self.x_input: batch_x, self.x_target: batch_x,
                                                        self.keep_prob: self.keep_param}) 

                if np.mod(it, control) == 0: 
                    
                    _, d_loss_curr = self.sess.run([discriminator_optimizer, self.dis_loss],
                        feed_dict={self.x_input: batch_x,
                        self.real_distribution: batch_z_real_dist,
                        self.keep_prob: self.keep_param})                     
                    
                else: 
                    
                    _, g_loss_curr = self.sess.run([generator_optimizer, self.generator_loss],
                        feed_dict={self.x_input: batch_x, self.keep_prob: self.keep_param}) 

                    
                _, d2_loss_curr = self.sess.run([discriminator2_optimizer, self.dis2_loss],
                        feed_dict={self.x_input: batch_x,
                        self.x_target: batch_x,
                        self.keep_prob: self.keep_param}) 
                                    
            self._is_train = False
            a_loss_epoch.append(a_loss_curr)
            d_loss_epoch.append(d_loss_curr)
            g_loss_epoch.append(g_loss_curr)
            d2_loss_epoch.append(d2_loss_curr)
            
            print(                
            "Epoch : [%d] ,  a_loss = %.4f, d_loss: %.4f ,  g_loss: %.4f, d2_loss: %.4f"
                                        % (ep, a_loss_curr, d_loss_curr, g_loss_curr, d2_loss_curr))    
       
        self.eval_cluster_on_test()

    # The autoencoder network
    def encoder(self, x, reuse=False):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """

        with tf.variable_scope('Encoder') as scope:
            if reuse:
                scope.reuse_variables()

            if self.is_bn:
                h = tf.layers.batch_normalization(

                    lrelu(dense(x, self.X_dim, self.g_h_dim[0], name='enc_h0_lin'), alpha=self.leak),
                    training=self._is_train, name='enc_bn0')
                    
                for i in range(1, self.num_layers):
                    h = tf.layers.batch_normalization(

                        lrelu(dense(h, self.g_h_dim[i - 1], self.g_h_dim[i], name='enc_h' + str(i) + '_lin'),
                              alpha=self.leak),
                        training=self._is_train, name='enc_bn' + str(i))                    

                z_post_m = dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_m' + str(self.num_layers) + '_lin')                
                z_post_v = tf.exp(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_v' + str(self.num_layers) + '_lin'))              
                
                h = tf.nn.relu(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_h' + str(self.num_layers) + '_lin'))

                l_post_m = dense(h, self.z_dim, 1, name='enc_l_post_m' + str(self.num_layers) + '_lin')                            
                l_post_v = tf.exp(dense(h, self.z_dim, 1, name='enc_l_post_v' + str(self.num_layers) + '_lin')) 
                

            else:

                h = tf.nn.dropout(lrelu(dense(x, self.X_dim, self.g_h_dim[0], name='enc_h0_lin'), alpha=self.leak),
                                  keep_prob=self.keep_prob)                
                
                for i in range(1, self.num_layers):
                    
                    h = tf.nn.dropout(lrelu(dense(h, self.g_h_dim[i - 1], self.g_h_dim[i], name='enc_h' + str(i) + '_lin'),
                              alpha=self.leak), keep_prob=self.keep_prob)                    

                z_post_m = dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_m' + str(self.num_layers) + '_lin')                
                z_post_v = tf.exp(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_v' + str(self.num_layers) + '_lin'))
                
                
                h = tf.nn.relu(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_h' + str(self.num_layers) + '_lin'))
                          

                l_post_m = dense(h, self.z_dim, 1, name='enc_l_post_m' + str(self.num_layers) + '_lin')                             
                l_post_v = tf.exp(dense(h, self.z_dim, 1, name='enc_l_post_v' + str(self.num_layers) + '_lin'))                                          
                            
            return h, z_post_m, z_post_v, l_post_m, l_post_v


    def decoder(self, z, reuse=False):
        """
        Decoder part of the autoencoder.
        :param z: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        with tf.variable_scope('Decoder') as scope:
            if reuse:
                scope.reuse_variables()

            if self.is_bn:

                h = tf.layers.batch_normalization(
                  
                    lrelu(dense(z, self.z_dim, self.g_h_dim[self.num_layers-1], name='dec_h' + str(self.num_layers-1) + '_lin'),
                          alpha=self.leak),                   
                    training=self._is_train, name='dec_bn' + str(self.num_layers-1))
                for i in range(self.num_layers-2, -1,-1):
                    h = tf.layers.batch_normalization(

                        lrelu(dense(h, self.g_h_dim[i + 1], self.g_h_dim[i], name='dec_h' + str(i) + '_lin'),
                             alpha=self.leak),                        
                        training=self._is_train, name='dec_bn' + str(i))
            else:
                h = tf.nn.dropout(lrelu(dense(z, self.z_dim, self.g_h_dim[self.num_layers-1], name='dec_h' + str(self.num_layers-1) + '_lin'),
                                        alpha=self.leak),                                  
                                  keep_prob=self.keep_prob)
                for i in range(self.num_layers-2, -1, -1):
                    h = tf.nn.dropout(
                        lrelu(dense(h, self.g_h_dim[i + 1], self.g_h_dim[i], name='dec_h' + str(i) + '_lin'),
                              alpha=self.leak), keep_prob=self.keep_prob)
               
            return h



    def discriminator(self, z, z_dim, reuse=False):    
        """
        Discriminator that is used to match the posterior distribution with a given prior distribution.
        :param z: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """
        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            if self.is_bn:

                h = tf.layers.batch_normalization(
                    lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='dis_h' + str(self.num_layers-1) + '_lin'),      
                          alpha=self.leak),
                    training=self._is_train, name='dis_bn' + str(self.num_layers-1))
                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.layers.batch_normalization(
                        lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis_h' + str(i) + '_lin'),
                              alpha=self.leak),
                        training=self._is_train, name='dis_bn' + str(i))

            else:

                h = tf.nn.dropout(
                    lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='dis_h' + str(self.num_layers-1) + '_lin'),
                          alpha=self.leak),
                    keep_prob=self.keep_prob)
                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.nn.dropout(
                        lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis_h' + str(i) + '_lin'),
                              alpha=self.leak), keep_prob=self.keep_prob)

            output = dense(h, self.d_h_dim[0], 1, name='dis_output')
            return output

    def discriminator2(self, z, z_dim, reuse=False):    
        """
        Discriminator that is used to match the posterior distribution with a given prior distribution.
        :param z: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """
        with tf.variable_scope('Discriminator2') as scope:
            if reuse:
                scope.reuse_variables()

            if self.is_bn:

                h = tf.layers.batch_normalization(
                    lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='dis2_h' + str(self.num_layers-1) + '_lin'),      
                          alpha=self.leak),
                    training=self._is_train, name='dis2_bn' + str(self.num_layers-1))
                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.layers.batch_normalization(
                        lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis2_h' + str(i) + '_lin'),
                              alpha=self.leak),
                        training=self._is_train, name='dis2_bn' + str(i))

            else:

                h = tf.nn.dropout(
                    lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='dis2_h' + str(self.num_layers-1) + '_lin'),
                          alpha=self.leak),
                    keep_prob=self.keep_prob)
                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.nn.dropout(
                        lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis2_h' + str(i) + '_lin'),
                              alpha=self.leak), keep_prob=self.keep_prob)

            output = dense(h, self.d_h_dim[0], 1, name='dis2_output')
            return output


    @property
    def model_dir(self):
        s = "DRA_{}_{}_b_{}_g{}_d{}_{}_{}_lr_{}_b1_{}_leak_{}_keep_{}_z_{}_{}_bn_{}_lam_{}_giter_{}_epoch_{}".format(
            datetime.datetime.now(), self.dataset_name, 
            self.batch_size, self.g_h_dim, self.d_h_dim, self.gen_activation, self.trans, self.lr, 
            self.beta1, self.leak, self.keep_param, self.z_dim, self.sampler, self.is_bn,
            self.lam, self.g_iter, self.epoch) 
        s = s.replace('[', '_')
        s = s.replace(']', '_')
        s = s.replace(' ', '')
        return s

    def sample_Z(self, m, n, sampler='uniform'):
        if self.sampler == 'uniform':
            return np.random.uniform(-1., 1., size=[m, n])
        elif self.sampler == 'normal':
            return np.random.randn(m, n)

    def next_batch(self, data, max_size):
        indx = np.random.randint(max_size - self.batch_size)
        return data[indx:(indx + self.batch_size), :]

    def sample_gaussian(self, mean, variance, scope=None):

        with tf.variable_scope(scope, 'sample_gaussian'):
            sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(variance))
            sample.set_shape(mean.get_shape())
            return sample

    # Zero-inflated negative binomial (ZINB) model is for modeling count variables with excessive zeros and it is usually for overdispersed count outcome variables.
    def zinb_model(self, x, mean, inverse_dispersion, logit, eps=1e-8): 
                                          
        expr_non_zero = - tf.nn.softplus(- logit) \
                        + tf.log(inverse_dispersion + eps) * inverse_dispersion \
                        - tf.log(inverse_dispersion + mean + eps) * inverse_dispersion \
                        - x * tf.log(inverse_dispersion + mean + eps) \
                        + x * tf.log(mean + eps) \
                        - tf.lgamma(x + 1) \
                        + tf.lgamma(x + inverse_dispersion) \
                        - tf.lgamma(inverse_dispersion) \
                        - logit 

        expr_zero = - tf.nn.softplus( - logit) \
                    + tf.nn.softplus(- logit + tf.log(inverse_dispersion + eps) * inverse_dispersion \
                                     - tf.log(inverse_dispersion + mean + eps) * inverse_dispersion) 
        
        template = tf.cast(tf.less(x, eps), tf.float32)
        expr =  tf.multiply(template, expr_zero) + tf.multiply(1 - template, expr_non_zero)
        return tf.reduce_sum(expr, axis=-1)


    def eval_cluster_on_test(self):

        # Embedding points in the test data to the latent space
        inp_encoder = self.data_test
        latent_matrix = self.sess.run(self.z, feed_dict={self.x_input: inp_encoder, self.keep_prob: 1.0})
       
        labels = self.labels_test
        K = np.size(np.unique(labels))        
        kmeans = KMeans(n_clusters=K, random_state=0).fit(latent_matrix)
        y_pred = kmeans.labels_

        print('Computing NMI ...')
        NMI = nmi(labels.flatten(), y_pred.flatten())
        print('Done !')

        print('NMI = {}'. 
              format(NMI)) 

        if not os.path.exists('Res_DRA/tune_logs'):
            os.makedirs('Res_DRA/tune_logs')

        out_file_name = 'Res_DRA/tune_logs/Metrics_{}.txt'.format(self.dataset_name)
        f = open(out_file_name, 'a')

        f.write('\n{}, NMI = {}'. 
                format(self.model_dir, NMI)) 
        
        f.close()

if __name__=='__main__':

    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.001]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.9]")
    flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
    flags.DEFINE_integer("z_dim", 10, "Latent space dimension")
    flags.DEFINE_integer("n_l", 2, "# Hidden Layers")
    flags.DEFINE_integer("g_h_l1", 256, "#Generator Hidden Units in Layer 1")
    flags.DEFINE_integer("g_h_l2", 256, "#Generator Hidden Units in Layer 2")
    flags.DEFINE_integer("g_h_l3", 0, "#Generator Hidden Units in Layer 3")
    flags.DEFINE_integer("g_h_l4", 0, "#Generator Hidden Units in Layer 4")
    flags.DEFINE_integer("d_h_l1", 256, "#Discriminator Hidden Units in Layer 1")
    flags.DEFINE_integer("d_h_l2", 256, "#Discriminator Hidden Units in Layer 2")
    flags.DEFINE_integer("d_h_l3", 0, "#Discriminator Hidden Units in Layer 3")
    flags.DEFINE_integer("d_h_l4", 0, "#Discriminator Hidden Units in Layer 4")
    flags.DEFINE_string("actv", "sig", "Decoder Activation [sig, tanh, lin]")
    flags.DEFINE_float("leak", 0.2, "Leak factor")
    flags.DEFINE_float("keep", 1.0, "Keep prob")
    flags.DEFINE_string("trans", "sparse", "Data Transformation [dense, sparse]")
    flags.DEFINE_string("dataset", "10x_73k", "The name of dataset [mnist, 10x_73k, 10x_68k, Zeisel, Macosko]")
    flags.DEFINE_string("checkpoint_dir", "/data/eugene/AAE-20180306-Hemberg/test_checkpoint", "Directory name to save the checkpoints [checkpoint]") 
    flags.DEFINE_string("sample_dir", "test_samples", "Directory name to save the image samples [samples]")
    flags.DEFINE_string("result_dir", "test_result", "Directory name to results of gene imputation [result]")
    flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
    flags.DEFINE_integer("g_iter", 2, "# Generator Iterations [2]")
    flags.DEFINE_boolean("bn", False, "True for batch Norm [False]")
    flags.DEFINE_float("lam", 10.0, "Lambda for regularization")
    flags.DEFINE_string("sampler", "normal", "The sampling distribution of z [uniform, normal, mix_gauss]")
    flags.DEFINE_string("model", "aae", "Model to train [aae, van_ae] [aae]")
    flags.DEFINE_integer("X_dim", 720, "Input dimension") 
    FLAGS = flags.FLAGS
    
    print ("dataset: {}".format(FLAGS.dataset)) 
    print ("checkpoint_dir: {}".format(FLAGS.checkpoint_dir)) 
    print ("n_l: {}".format(FLAGS.n_l)) 
    print ("g_h_l1: {}".format(FLAGS.g_h_l1)) 
    print ("g_h_l2: {}".format(FLAGS.g_h_l2)) 
    print ("g_h_l3: {}".format(FLAGS.g_h_l3)) 
    print ("g_h_l4: {}".format(FLAGS.g_h_l4))
    print ("d_h_l1: {}".format(FLAGS.d_h_l1)) 
    print ("d_h_l2: {}".format(FLAGS.d_h_l2)) 
    print ("d_h_l3: {}".format(FLAGS.d_h_l3)) 
    print ("d_h_l4: {}".format(FLAGS.d_h_l4)) 
    print ("batch_size: {}".format(FLAGS.batch_size)) 
    print ("beta1: {}".format(FLAGS.beta1)) 
    print ("learning_rate: {}".format(FLAGS.learning_rate)) 
    print ("z_dim: {}".format(FLAGS.z_dim)) 
    print ("epoch: {}".format(FLAGS.epoch)) 
    print ("leak: {}".format(FLAGS.leak)) 
    print ("keep: {}".format(FLAGS.keep)) 
    print ("model: {}".format(FLAGS.model)) 
    print ("trans: {}".format(FLAGS.trans)) 
    print ("actv: {}".format(FLAGS.actv)) 
    print ("X_dim: {}".format(FLAGS.X_dim)) 
    print ("bn: {}".format(FLAGS.bn)) 
    print ("g_iter: {}".format(FLAGS.g_iter)) 
    print ("lam: {}".format(FLAGS.lam)) 
    print ("sampler: {}".format(FLAGS.sampler))    


    def main(_):

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction=0.333
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:

            g_h_dim = [FLAGS.g_h_l1, FLAGS.g_h_l2, FLAGS.g_h_l3, FLAGS.g_h_l4]
            d_h_dim = [FLAGS.d_h_l1, FLAGS.d_h_l2, FLAGS.d_h_l3, FLAGS.d_h_l4]

            if FLAGS.model == 'dra':
                    test_dra = Test_DRA(
                        sess,
                        epoch=FLAGS.epoch,
                        lr=FLAGS.learning_rate,
                        beta1=FLAGS.beta1,
                        batch_size=FLAGS.batch_size,
                        X_dim=FLAGS.X_dim, 
                        z_dim=FLAGS.z_dim,
                        dataset_name=FLAGS.dataset,
                        checkpoint_dir=FLAGS.checkpoint_dir,
                        sample_dir=FLAGS.sample_dir,
                        result_dir = FLAGS.result_dir,
                        num_layers=FLAGS.n_l,
                        g_h_dim=g_h_dim[:FLAGS.n_l],
                        d_h_dim=d_h_dim[:FLAGS.n_l],
                        gen_activation=FLAGS.actv,
                        leak = FLAGS.leak,
                        keep_param=FLAGS.keep,
                        trans=FLAGS.trans,
                        is_bn=FLAGS.bn,
                        g_iter=FLAGS.g_iter,
                        lam=FLAGS.lam,
                        sampler=FLAGS.sampler)

            # show_all_variables()
            if FLAGS.train:
                if FLAGS.model == 'dra':
                    test_dra.train_cluster()
                    

    tf.app.run()




    
