import tensorflow as tf
# from BatchIterator import BatchIterator # Batch iterator is tensorflow-based
import pathlib
import random
import numpy as np
from time import time


# think about
# session management (session should come from colab)
# is the optimizer included in the model class? (sklearn style)
# logging for tensorboard (summaries)
class BetaVAE:
    def __init__(self,
                 img_side=128,
                 batch_size=10,
                 epochs=10,
                 n_z=10,
                 beta=1.3,
                 lr=1e-3,
                 epsilon=1e-8):
        self.AUTOTUNE   = tf.data.experimental.AUTOTUNE
        self.img_side   = img_side
        self.img_size   = self.img_side**2
        self.batch_size = batch_size
        self.epochs     = epochs
        self.n_z        = n_z
        self.beta       = beta
        self.lr         = lr
        self.epsilon    = epsilon

        tf.reset_default_graph()

        self.input = tf.placeholder(tf.float32,
                                    [None, self.img_size],
                                    name="Inputs")
        self.latent = tf.placeholder(tf.float32,
                                     [None, self.n_z],
                                     name="Latents")

        self.encoder(self.input)
        self.decoder(self.z)
        # self.output = self.model
        self.gen_img = self.decoder(self.latent)

        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        self.recon_loss = -tf.reduce_sum(self.input
                                         * tf.log(self.epsilon + self.model)
                                         + (1 - self.input)
                                         * tf.log(self.epsilon + 1-self.model),
                                         axis=1)
        self.recon_loss = tf.reduce_mean(self.recon_loss)

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between
        # the latent distribution and N(0, 1)
        self.latent_loss = -0.5 * tf.reduce_sum(
                            1 + self.log_sigma_sq - tf.square(self.mu) -
                            tf.exp(self.log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(self.latent_loss)

        # Total loss
        self.loss = self.recon_loss + self.beta * self.latent_loss
        self.plottableLoss = []
        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def fit(self, sess, path):
        self.load_data(path)
        # Create Tensorflow Iterator
        path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        ds = path_ds.map(self.load_and_preprocess_image,
                         num_parallel_calls=self.AUTOTUNE)
        ds = ds.shuffle(buffer_size=self.batch_size * 2)
        ds = ds.repeat(self.data_repeats)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        self.iterator = ds.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

        print("Begin Session: Processing {:d} images in {:d} batches".format(
              self.batch_size * self.n_batches * self.epochs, self.n_batches))
        # Session
        sess.run(tf.global_variables_initializer())
        sess.run(self.iterator.initializer)
        self.err = False
        for epoch in range(self.epochs):
            tic = time()
            try:
                print("Epoch {:>2}:  ".format(epoch), end="")
                for i in range(self.n_batches - 2):
                    print(".", end="")
                    X = self.get_next(sess)
                    sess.run(self.opt, feed_dict={input: X})
                    tloss = sess.run(self.loss, feed_dict={input: X})
                    if np.isnan(tloss) or np.isinf(tloss):
                        print("Error: loss went cray cray")
                        self.err = True
                        break
                    self.plottableLoss.append(tloss)
                if self.err:
                    break
                print("\nLoss= {:>8.4f} Time: {:>8.4f}".format(tloss,
                                                               time() - tic))
            except tf.errors.OutOfRangeError:
                pass

    def predict(self, sess, x):
        return sess.run(self.model, feed_dict={input:x})

    def get_next(self, sess):
        return sess.run(self.next_element)

    def encoder(self, x):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            # dim = x.get_shape().as_list() # https://datascience.stackexchange.com/questions/30810/gan-with-conv2d-using-tensorflow-shape-error
            # x = tf.reshape(x, shape = [-1, *dim[1:]])
            self.model = tf.reshape(x, [-1, self.img_side, self.img_side, 1])
            self.model = tf.layers.Conv2D(filters=6 , kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(self.model)
            self.model = tf.layers.Conv2D(filters=12, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(self.model)
            self.model = tf.layers.Conv2D(filters=24, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(self.model)
            self.model = tf.layers.Flatten()(self.model)
            self.mu = tf.layers.Dense(self.n_z)(self.model)
            self.log_sigma_sq = tf.layers.Dense(self.n_z)(self.model)
            self.eps = tf.random_normal(shape=tf.shape(self.log_sigma_sq),
                                        mean=0, stddev=1, dtype=tf.float32)
            self.z = self.mu + tf.sqrt(tf.exp(self.log_sigma_sq)) * self.eps
            # return z, log_sigma_sq, mu

    def decoder(self, x):
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            self.model = tf.layers.Dense(7 * 7 * 32)(x)
            self.model = tf.reshape(self.model, [-1, 7, 7, 32])
            self.model = tf.layers.Conv2DTranspose(filters=24, kernel_size=3, strides=(2, 2), padding="same", activation=tf.nn.relu)(self.model)
            self.model = tf.layers.Conv2DTranspose(filters=12, kernel_size=3, strides=(2, 2), padding="same", activation=tf.nn.relu)(self.model)
            self.model = tf.layers.Conv2DTranspose(filters=6 , kernel_size=3, strides=(2, 2), padding="same", activation=tf.nn.relu)(self.model)
            self.model = tf.layers.Flatten()(self.model)
            self.model = tf.layers.Dense(self.img_size,
                                         activation=tf.nn.sigmoid)(self.model)
            # return self.model

    def load_data(self, path, format="png"):
        self.data_root = pathlib.Path(path)
        self.all_image_paths = list(self.data_root.glob("*." + format))
        self.all_image_paths = [str(path) for path in self.all_image_paths]
        random.shuffle(self.all_image_paths)
        self.image_count = len(self.all_image_paths)
        self.data_repeats = self.epochs * 2
        self.n_batches = self.image_count // self.batch_size

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize_images(image, [self.img_side, self.img_side])
    #     image = tf.image.per_image_standardization(image)
    #     image = image + tf.reduce_min(image)
        image /= tf.reduce_max(image)
        return image

    def load_and_preprocess_image(self, path):
        image = tf.read_file(path)
        return self.preprocess_image(image)
