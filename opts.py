import tensorflow as tf


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


# Leaky Relu
def lrelu(x, alpha = 0.2, name='lrelu'):
    return tf.maximum(x, alpha*x)

def dense(x, inp_dim, out_dim, name = 'dense'):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param inp_dim: no. of input neurons
    :param out_dim: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, out_dim]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[inp_dim, out_dim],
                                  initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", shape=[out_dim], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out