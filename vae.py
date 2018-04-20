import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from os.path import join
#from joblib import Parallel, delayed
#from functools import partial


def lrelu(x, alpha=0.1, name='leaky_relu'):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def parametric_relu(x, name='param_relu'):
    alphas = tf.get_variable('alpha', x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5
    return pos + neg



def normalize_batch(data, normalize="median"):    
    if normalize == "median":        
        data = data - np.median(data, axis=1, keepdims=True)
        norm = np.amax(np.absolute(data), axis=1, keepdims=True)
        norm[norm<=0.0] = 1.0
        #norm = np.percentile(data, 75, axis=1, keepdims=True) 
        #norm -= np.percentile(data, 25, axis=1, keepdims=True)
        #norm[norm<=0.0] = 1.0
        data = data/norm
    elif normalize == "minmax":
        min_data = np.amin(data, axis=1, keepdims=True)
        max_data = np.amax(data, axis=1, keepdims=True)
        norm = max_data - min_data
        norm[norm<=0.0] = 1.0
        data = (data - min_data)/norm
    elif normalize == "standard":
        data = data - np.mean(data, axis=1, keepdims=True)
        norm = np.std(data, axis=1, keepdims=True)
        norm[norm<=0.0] = 1.0
        data = data/norm
    return data.astype('float32')

class Autoencoder(object):
    
    def add_conv_layer(self, x, filters=32, kernel_size=3, stride=1, padding='SAME', scope='scope', 
                       activation=tf.nn.relu, dropout=True, batch_norm=True, transpose=False):
        if not transpose:
            x = layers.conv2d(inputs=x, num_outputs=filters, kernel_size=kernel_size, stride=stride, padding=padding,
                              activation_fn=activation, weights_initializer=layers.xavier_initializer(dtype=tf.float32),
                              biases_initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                              scope=scope)
        else:
            x = layers.conv2d_transpose(inputs=x, num_outputs=filters, kernel_size=kernel_size, stride=stride, padding=padding,
                              activation_fn=activation, weights_initializer=layers.xavier_initializer(dtype=tf.float32),
                              biases_initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                              scope=scope)
        if self.debug:
            print(x.shape)
        if batch_norm and self.use_batch_norm:
            x = layers.batch_norm(inputs=x, is_training=self.is_training, decay=0.9, center=False, scale=False, 
                                  fused=True, epsilon=1e-3, zero_debias_moving_mean=True, scope=scope+'_BN')
        if dropout and self.keep_prob < 1.0:
            x = layers.dropout(inputs=x, keep_prob=self.keep_prob, is_training=self.is_training, scope=scope+'_drop')
        return x
    
    def add_fc_layer(self, x, neurons=32, activation=tf.nn.relu, scope='scope', dropout=True, batch_norm=True):
        x = layers.fully_connected(inputs=x, num_outputs=neurons, activation_fn=activation, 
                                   weights_initializer=layers.xavier_initializer(dtype=tf.float32), 
                                   biases_initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                                   scope=scope)
        if self.debug:
            print(x.shape)
        if batch_norm and self.use_batch_norm:
            x = layers.batch_norm(inputs=x, is_training=self.is_training, decay=0.9, center=False, scale=False, 
                                  fused=True, epsilon=1e-3, zero_debias_moving_mean=True, scope=scope+'_BN')
        if dropout and self.keep_prob < 1.0:
            x = layers.dropout(inputs=x, keep_prob=self.keep_prob, is_training=self.is_training, scope=scope+'_drop')
        return x
    
    def build_encoder(self, n_filters=32, arch=1):        
        if arch == 1: # Convolutional
            encoder_h = tf.reshape(self.input_data, [-1, self.npix, self.npix, 1])
            encoder_h = self.add_conv_layer(encoder_h, filters=n_filters//4, stride=2, scope='conv1_st')
            encoder_h = self.add_conv_layer(encoder_h, filters=n_filters//2, stride=2, scope='conv2_st')
            encoder_h = layers.flatten(encoder_h)            
        elif arch == 2: # Fully-connected
            encoder_h = self.add_fc_layer(self.input_data, neurons=n_filters, scope='hidden')
        # Output of the encoder
        self.encoder_mu = self.add_fc_layer(encoder_h, neurons=self.latent_dim, 
                                            activation=None, batch_norm=False, dropout=False, scope='latent_mu')        
        if self.variational:
            self.encoder_lv = self.add_fc_layer(encoder_h, neurons=self.latent_dim, 
                                                activation=None, batch_norm=False, dropout=False, scope='latent_lv')
            
    def build_decoder(self, n_filters=32, arch=1, resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR ):        
        
        if not self.variational:
            decoder_input = self.encoder_mu
        else:
            decoder_input = self.z
        decoder_input = tf.reshape(decoder_input, [-1, self.latent_dim])
        
        if arch == 1:  # Convolutional Decoder       
            sub_pix_dim = self.npix//4+1            
            decoder_h = self.add_fc_layer(decoder_input, neurons=sub_pix_dim*sub_pix_dim*n_filters//8, scope='hidden')
            decoder_h = tf.reshape(decoder_h, [-1, sub_pix_dim, sub_pix_dim, n_filters//8])
            
            decoder_h = self.add_conv_layer(decoder_h, filters=n_filters//2, scope='conv2')   
            decoder_h = tf.image.resize_images(decoder_h, [self.npix//2+1, self.npix//2+1],
                                               method=resize_method)
            decoder_h = self.add_conv_layer(decoder_h, filters=n_filters//4, scope='conv3')   
            decoder_h = tf.image.resize_images(decoder_h, [self.npix, self.npix], 
                                               method=resize_method)
            # Decoder output
            decoder_mu = self.add_conv_layer(decoder_h, filters=1, activation=None, 
                                             dropout=False, batch_norm=False, scope='decoder_mu')
        elif arch == 2:  # MLP decoder
            decoder_h = self.add_fc_layer(decoder_input, neurons=n_filters, scope='hidden') 
            decoder_mu = self.add_fc_layer(decoder_h, neurons=self.npix*self.npix, activation=None,
                                           dropout=False, batch_norm=False, scope='decoder_mu')
            
        # Fix shape
        self.decoder_mu = tf.reshape(decoder_mu, [-1, self.mc_realizations, self.npix*self.npix])
        if self.variational:
            decoder_lv = self.add_fc_layer(decoder_h, neurons=1, activation=tf.nn.tanh,
                                           dropout=False, batch_norm=False, scope='decoder_lv') 
            self.decoder_lv = tf.reshape(2*decoder_lv, [-1, self.mc_realizations, 1])

    def __init__(self, batch_size=128, latent_dim=21, npix=21, arch=1, 
                 variational=False, importance=False, mc_realizations=1, 
                 keep_prob=1.0, use_batch_norm=False, 
                 n_filters=32, kernel_size=3,
                 log_dir="/tmp/tensorboard/", tag=None, rseed=0):
        tf.reset_default_graph()
        np.random.seed(rseed)
        tf.set_random_seed(rseed)
        self.variational = variational
        self.importance = importance
        if self.variational:
            self.mc_realizations = mc_realizations
        else:
            self.mc_realizations = 1
        #self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.npix = npix
        self.keep_prob = keep_prob  # For dropout
        if keep_prob < 1.0:
            print("Using dropout in all layers")
        self.arch = arch
        print("Using arch %d" %(arch))
        self.input_dim = self.npix*self.npix
        self.use_batch_norm = use_batch_norm  # General switch to use BN        
        self.debug = True
        self.input_data = tf.placeholder(tf.float32, [None, self.input_dim], name='input_data')
        self.is_training = tf.placeholder(tf.bool, name='is_training')        
        #self.global_step = 0 
        #self.global_step = tf.Variable(0, trainable=False)
        self.reg_param = tf.placeholder(tf.float32, name='reg_param')
        
        if self.variational:
            if importance:
                name = "IWAE"
            else:
                name = "VAE"
            name = name+"_MC"+str(mc_realizations)
        else:
            name = "AE"
        name = name+"_L"+str(self.latent_dim)+"_npix"+str(self.npix)
        
        if self.use_batch_norm:
            name = name+"_BN"
        if self.keep_prob < 1.0:
            name = name+"_DO"
        name = name+"_ARCH"+str(self.arch)
        name = name+"_Nf"+str(n_filters)
        if tag is not None:
            name = name+"_"+tag
        self.name = name        
        
        with tf.variable_scope('encoder'):
            self.build_encoder(arch=arch, n_filters=n_filters)

        if self.variational:
            with tf.variable_scope('sampling'):
                batch_size_tf = tf.shape(self.encoder_mu)[0]
                epsilon = tf.random_normal([batch_size_tf, mc_realizations, latent_dim], 
                                           0.0, 1.0, dtype=tf.float32, name='epsilon')
                self.z = tf.add(tf.expand_dims(self.encoder_mu, axis=[1]), 
                                tf.multiply(tf.expand_dims(tf.exp(0.5*self.encoder_lv), axis=[1]), 
                                                           epsilon), name='z') 
        else:
            var = [v for v in tf.trainable_variables() if v.name == "encoder/latent_mu/weights:0"]
            print("Applying L1 regularization to:")
            print(var)
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1.0, scope=None)
            regularization_penalty = -tf.contrib.layers.apply_regularization(l1_regularizer, var)
            
                    
        with tf.variable_scope('decoder'):
            self.build_decoder(arch=arch, n_filters=n_filters)
        
        log2pi = tf.log(tf.constant(2.0*np.pi, dtype=tf.float32))
       
        if self.variational:
            with tf.variable_scope('log_pxz'):
                logpxz = -0.5*tf.reduce_sum(tf.divide(tf.square(tf.subtract(tf.expand_dims(self.input_data, axis=[1]),
                                                                            self.decoder_mu)),
                                                      tf.exp(self.decoder_lv)) + (self.decoder_lv) + log2pi, axis=[2])                
            
            with tf.variable_scope('log_qzx'):
                if not importance:
                    logqzx = tf.expand_dims(0.5*tf.reduce_sum(tf.square(self.encoder_mu) + tf.exp(self.encoder_lv) \
                                                              - 1. - self.encoder_lv, axis=[1]), axis=[1])                    
                else:
                    logqzx = 0.5*tf.reduce_sum(tf.square(self.z)  \
                                               - tf.divide(tf.square(tf.subtract(self.z, 
                                                                                 tf.expand_dims(self.encoder_mu, axis=[1]))), \
                                                           tf.exp(tf.expand_dims(self.encoder_lv, axis=[1]))) \
                                               - tf.expand_dims(self.encoder_lv, axis=[1]), axis=[2])
            with tf.variable_scope('ELBO'):
                if not importance:
                    self.ELBO = tf.reduce_mean(tf.subtract(logpxz, logqzx), axis=[0, 1])
                else:
                    self.ELBO = tf.reduce_mean(tf.reduce_logsumexp(tf.subtract(logpxz, logqzx), axis=[1]) \
                                               -tf.log(tf.cast(self.mc_realizations, tf.float32)), axis=[0])                    
        else:
            with tf.variable_scope('log_pxz'):
                logpxz = logpxz = -0.5*tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(self.input_data, axis=[1]),
                                                                           self.decoder_mu)), axis=[2])
            self.ELBO = tf.reduce_mean(logpxz, axis=[0]) + self.reg_param*regularization_penalty
            
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -10., 10.)

        with tf.variable_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=1e-4, epsilon=1e-6)
                print(optimizer)
                tvars = tf.trainable_variables()
                # We want to maximize the evidence lower bound
                grads_and_vars = optimizer.compute_gradients(-self.ELBO, tvars) 
                clipped = [(ClipIfNotNone(grad), tvar) for grad, tvar in grads_and_vars]
                self.train_op = optimizer.apply_gradients(grads_and_vars, name="maximize_ELBO")

        logpxz_hist = tf.summary.scalar("logpxz_train", tf.reduce_mean(logpxz, axis=[0, 1]))
        logpxz_hist_val = tf.summary.scalar("logpxz_val", tf.reduce_mean(logpxz, axis=[0, 1]))        
        if self.variational:
            logqzx_hist = tf.summary.scalar("logqzx_train", tf.reduce_mean(logqzx, axis=[0, 1]))
            logqzx_hist_val = tf.summary.scalar("logqzx_val", tf.reduce_mean(logqzx, axis=[0, 1]))
            self.merged_summaries_train = tf.summary.merge([logpxz_hist, logqzx_hist], name='Summary_train')
            self.merged_summaries_val = tf.summary.merge([logpxz_hist_val, logqzx_hist_val], name='Summary_val')
        else:
            self.merged_summaries_train = tf.summary.merge([logpxz_hist], name='Summary_train')
            self.merged_summaries_val = tf.summary.merge([logpxz_hist_val], name='Summary_val')        
        
        tf.add_to_collection("my_handles", self.input_data)
        tf.add_to_collection("my_handles", self.is_training)
        tf.add_to_collection("my_handles", self.ELBO)
        tf.add_to_collection("my_handles", self.train_op)
        tf.add_to_collection("my_handles", self.encoder_mu)
        tf.add_to_collection("my_handles", self.decoder_mu)
        if variational:
            tf.add_to_collection("my_handles", self.encoder_lv)
            tf.add_to_collection("my_handles", self.z)
            tf.add_to_collection("my_handles", self.decoder_lv)
        
        self.init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.reset_parameters()
        if not log_dir is None:
            print("Saving data to %s dir" %(join(log_dir, self.name)))
            self.writer = tf.summary.FileWriter(join(log_dir, self.name), self.sess.graph)
        else:
            self.writer = None
        self.saver = tf.train.Saver(max_to_keep=100)
        
    def reconstruct(self, data):
        rdata_mu, rdata_lv = self.sess.run([self.decoder_mu, self.decoder_lv], 
                                           feed_dict={self.input_data: data, self.is_training: False}) 
        return rdata_mu, rdata_lv
    
    def generate(self, z):
        rdata_mu, rdata_lv = self.sess.run([self.decoder_mu, self.decoder_lv],
                                           feed_dict={self.z: z, self.is_training: False})
        return rdata_mu, rdata_lv
    
    def encode(self, data):
        if self.variational:
            code_mu, code_lv = self.sess.run([self.encoder_mu, self.encoder_lv],
                                         feed_dict={self.input_data: data, self.is_training: False})
            return code_mu, code_lv
        else:
            code_mu = self.sess.run(self.encoder_mu,
                                         feed_dict={self.input_data: data, self.is_training: False})
            return code_mu
        
    def reset_parameters(self):
        self.sess.run(self.init)
        self.global_step = 0
        
    def close(self):
        print("Closing session")
        self.writer.close()
        self.sess.close()
        
    def partial_fit(self, mini_batch, step, train=False, reg_param=None):
        if reg_param is None:
            reg_param = 1.0
        elif reg_param == "annealing":
            reg_param = np.amin([1.0, 1e-2 + (1.0-1e-2)*(step)/(10*4000)])
            
        if train:
            _, cost, summary_str = self.sess.run([self.train_op, self.ELBO, self.merged_summaries_train], 
                                                 feed_dict={self.input_data: mini_batch, 
                                                            self.is_training: True,
                                                            self.reg_param: reg_param})            
        else:
            cost, summary_str = self.sess.run([self.ELBO, self.merged_summaries_val], 
                                              feed_dict={self.input_data: mini_batch,
                                                         self.is_training: False,
                                                         self.reg_param: reg_param})
        if not self.writer is None:
            self.writer.add_summary(summary_str, global_step=step)
            self.writer.flush()
        return cost
    
    def get_number_of_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
    
    def save_network(self, step):
        self.saver.save(self.sess, "models/"+self.name+".ckpt", global_step=step)
        
    def load_network(self, model_path):
        tf.reset_default_graph()
        try:
            self.saver = tf.train.import_meta_graph(model_path+".meta")
        except OSError:
            print("File does not exist")
        self.saver.restore(self.sess, model_path)
        