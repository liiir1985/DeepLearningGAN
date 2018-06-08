import os
import math
import time
import sys
from glob import glob

import tensorflow as tf
from six.moves import xrange
import numpy as np
from utils import show_all_variables, get_image, save_images

batch_size = 64
learning_rate = 0.00015
learning_rate_d = 0.00015

beta1 = 0.5
epochs = 100
train_size = np.inf

# WGAN_GP parameter
lambd = 0.25       # The higher value, the more stable, but the slower convergence
disc_iters = 5

input_fname_pattern='*.jpg'

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def generator(z, output_width, output_height, training, reuse = None):
    with tf.variable_scope("generator") as scope:
        if reuse == True:
            scope.reuse_variables()
        gf_dim = 64
        s_h, s_w = output_height, output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        g_z_ = tf.layers.dense(z, gf_dim*8*s_h16*s_w16, activation = tf.nn.relu,kernel_initializer=tf.random_normal_initializer(stddev=0.02), reuse = reuse, name='g_z_')
        g_h0 = tf.reshape(
                g_z_, [-1, s_h16, s_w16, gf_dim * 8])
        g_h0 = tf.layers.batch_normalization(g_h0, training = training, reuse = reuse, name='g_h0')

        g_h1 = tf.layers.batch_normalization(
                    tf.layers.conv2d_transpose(g_h0, gf_dim * 4, 5, 2, padding='same', activation= tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name='g_h1/deconv2d'), training = training, reuse = reuse, name='g_h1/bn')
        g_h2 = tf.layers.batch_normalization(
                    tf.layers.conv2d_transpose(g_h1, gf_dim * 2, 5, 2, padding='same', activation= tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name='g_h2/deconv2d'), training = training, reuse = reuse, name='g_h2/bn')
        g_h3 = tf.layers.batch_normalization(
                    tf.layers.conv2d_transpose(g_h2, gf_dim, 5, 2, padding='same', activation= tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name='g_h3/deconv2d'), training = training, reuse = reuse, name='g_h3/bn')
        g_h4 = tf.layers.conv2d_transpose(g_h3, 3, 5, 2, padding='same', activation= tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name='g_h4/deconv2d')

    return g_h4

def discriminator(image, training, reuse = None):
    df_dim = 64
    tf
    d_h0 = tf.layers.batch_normalization(
                    tf.layers.conv2d(image, df_dim, 5, 2, padding='same', activation = tf.nn.leaky_relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name = 'd_h0'), training = training, reuse = reuse, name='d_h0/bn')
    d_h1 = tf.layers.batch_normalization(
                    tf.layers.conv2d(d_h0, df_dim * 2, 5, 2, padding='same', activation = tf.nn.leaky_relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name = 'd_h1/conv2d'), training = training, reuse = reuse, name='d_h1/bn')
    d_h2 = tf.layers.batch_normalization(
                    tf.layers.conv2d(d_h1, df_dim * 4, 5, 2, padding='same', activation = tf.nn.leaky_relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name = 'd_h2/conv2d'), training = training, reuse = reuse, name='d_h2/bn')
    d_h3 = tf.layers.batch_normalization(
                    tf.layers.conv2d(d_h2, df_dim * 8, 5, 2, padding='same', activation = tf.nn.leaky_relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), reuse = reuse, name = 'd_h3/conv2d'), training = training, reuse = reuse, name='d_h3/bn')

    d_h4 = tf.layers.dense(tf.reshape(d_h3, [-1,4*4*512]),1,kernel_initializer=tf.random_normal_initializer(stddev=0.02), reuse = reuse, name = 'd_h4/fc')

    return tf.nn.sigmoid(d_h4), d_h4

def sampler(z, output_width, output_height):
    return generator(z, output_width, output_height, False, True)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise



def save(saver, sess, checkpoint_dir, step):
    model_name = "DCGAN.model"
    #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    saver.save(sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

def run(sess, training = True, z_dim = 100, image_dims=[64,64,3]):
    inputs = tf.placeholder(
      tf.float32, [None] + image_dims, name='real_images')

    z = tf.placeholder(
      tf.float32, [None, z_dim], name='z')
    z_sum = tf.summary.histogram("z", z)

    G                  = generator(z,64, 64, True)
    D, D_logits   = discriminator(inputs, True)
    sampler_            = sampler(z, 64, 64)
    D_, D_logits_ = discriminator(G, True, reuse=True)
    
    d_sum = tf.summary.histogram("d", D)
    d__sum = tf.summary.histogram("d_", D_)
    G_sum = tf.summary.image("G", G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    """ DCGAN
    d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
    g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))
                          
    d_loss = d_loss_real + d_loss_fake"""

     # get loss for discriminator
    d_loss_real = tf.reduce_mean(tf.scalar_mul(-1,D_logits))
    d_loss_fake = tf.reduce_mean(D_logits_)

    d_loss = d_loss_real + d_loss_fake
    # get loss for generator
    g_loss = tf.reduce_mean(tf.scalar_mul(-1,D_logits_))

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    
    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    """
    alpha = tf.random_uniform( [batch_size] + image_dims, minval=0.,maxval=1.)
    differences = G - inputs # This is different from MAGAN
    interpolates = inputs + (alpha * differences)
    _,D_inter, = discriminator(interpolates, training=True, reuse=True)
    gradients = tf.gradients(D_inter, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    d_loss += lambd * gradient_penalty
    """

    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)

    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    # Clip D's variables ----WGAN
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]


    saver = tf.train.Saver()

    show_all_variables()

    d_learning_rate = tf.placeholder(tf.float32)
    
    if(training):
        """ ADAM Optimizer
        """
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        d_optimizer = tf.train.AdamOptimizer(learning_rate_d, beta1=beta1)
        with tf.control_dependencies(extra_update_ops):
            d_optim = d_optimizer.minimize(d_loss, var_list=d_vars)


        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        with tf.control_dependencies(extra_update_ops):
            g_optim = g_optimizer.minimize(g_loss, var_list=g_vars)
        

        """
        d_optim = tf.train.RMSPropOptimizer(learning_rate = learning_rate) \
                .minimize(d_loss, var_list=d_vars)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        g_optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)        
        with tf.control_dependencies(extra_update_ops):
            g_optim = g_optimizer.minimize(g_loss, var_list=g_vars)
        """

        try:
          tf.global_variables_initializer().run()
        except:
          tf.initialize_all_variables().run()

        g_sum = tf.summary.merge([z_sum, d__sum,
            G_sum, d_loss_fake_sum, g_loss_sum])
        d_sum = tf.summary.merge(
            [z_sum, d_sum, d_loss_real_sum, d_loss_sum])
        
        writer = tf.summary.FileWriter("./logs", sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(batch_size , z_dim))

        data = glob(os.path.join("./data", input_fname_pattern))

        sample_files = data[0:batch_size]
        sample = [
          get_image(sample_file) for sample_file in sample_files]
        
        sample_inputs = np.array(sample).astype(np.float32)

        could_load, counter = load(saver, sess, 'model')
        if could_load != True:
            counter = 1
        start_time = time.time()
        
        for epoch in xrange(epochs):
            data = glob(os.path.join(
                "./data", input_fname_pattern))
            #seed = 547
            #np.random.seed(seed)
            np.random.shuffle(data)
            batch_idxs = min(len(data), train_size) // batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*batch_size:(idx+1)*batch_size]
                batch = [
                      get_image(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch)

                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]) \
                    .astype(np.float32)

                for counter_d in xrange(0, disc_iters):                
                    # Update D network
                    _, summary_str, _ = sess.run([d_optim, d_sum, clip_D],
                        feed_dict={ inputs: batch_images, z: batch_z})
                    writer.add_summary(summary_str, counter)
                
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]) \
                .astype(np.float32)
                # Update G network
                _, summary_str = sess.run([g_optim, g_sum],
                    feed_dict={ z: batch_z, inputs: batch_images })
                writer.add_summary(summary_str, counter)

                """# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = sess.run([g_optim, g_sum],
                feed_dict={ z: batch_z, inputs: batch_images })
                writer.add_summary(summary_str, counter)"""
          
                errD_fake = d_loss_fake.eval({ z: batch_z })
                errD_real = d_loss_real.eval({ inputs: batch_images })
                errG = g_loss.eval({z: batch_z})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake+errD_real, errG))                

                counter += 1
                if np.mod(counter, batch_idxs) == 1:
                    try:
                      samples, d_loss_, g_loss_ = sess.run(
                        [sampler_, d_loss, g_loss],
                        feed_dict={
                            z: sample_z,
                            inputs: sample_inputs,
                        },
                      )
                      save_images(samples, (8,8), './{}/train_{:02d}_{:04d}_{:08d}.png'.format('samples', epoch, idx, counter))
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss_, g_loss_)) 
                    except:
                        e = sys.exc_info()[0]
                        print(e)

                if np.mod(counter, batch_idxs*10) == 1:
                  save(saver, sess, 'model', counter)
    else:
        could_load, counter = load(saver, sess, 'model')
        if could_load == True:
            sample_z = np.random.uniform(-1, 1, size=(batch_size//4 , z_dim))
            samples = sess.run(
                        [sampler_],
                        feed_dict={
                            z: sample_z,
                        },
                      )
            img = samples[0]
            save_images(samples[0], (4,4), './{}/run_{:06d}.png'.format('samples', counter))
        else:
            print('load model failed')
        return
def load(saver, sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

if __name__ == '__main__':
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        run(sess, True)
