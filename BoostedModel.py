import tensorflow as tf
# from BatchIterator import BatchIterator # Batch iterator is tensorflow-based
import pathlib
import random
import numpy as np
from time import time
import os

# think about
# session management (session should come from colab)
# is the optimizer included in the model class? (sklearn style)
# logging for tensorboard (summaries)
class BoostedBetaVAE:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    def __init__(self,
                 path="./dataset",
                 img_side=128,
                 batch_size=16,
                 epochs=10,
                 n_z=10,
                 beta=100,
                 lr=1e-3,
                 format="jpg",
                 extra_repeats=2,
                 n_channels=1,
                 epsilon=1e-10):
        self.img_side   = img_side
        self.img_size   = self.img_side**2
        self.batch_size = batch_size
        self.epochs     = epochs
        self.n_z        = n_z
        self.beta       = beta
        self.lr         = lr
        self.epsilon    = epsilon
        self.n_channels = n_channels

        # Build Model Architecture
        tf.reset_default_graph()

        # Data loading
        self.load_data(path, format=format, extra_repeats=extra_repeats)
        self.next_element = self.create_iterator()

        # Placeholders
        self.input = tf.placeholder(tf.float32,
                                    [None, self.img_size],
                                    name="Inputs")
        self.latent = tf.placeholder(tf.float32,
                                     [None, self.n_z],
                                     name="Latents")

        self.siameseLabel = tf.placeholder(tf.float32, [None, 1])

        # Encoder, decoder and sampling
        self.encode(self.input) # z, mu, and lss are saved in the object.
        self.model = self.decode(self.z)

        # siamese network
        self.compScore = self.siamese(self.input)

        self.sample = self.decode(self.latent)

        # LOSSES
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


        # siamese loss
        self.siameseLoss = tf.nn.l2_loss(self.siameseLabel - self.compScore)

        # Total loss
        self.loss = self.recon_loss + self.beta * self.latent_loss
        self.plottableLoss = []

        # Optimizer Siamese
        siaVarList = [x for x in tf.global_variables() if x.name.startswith('siamese_')]
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.siaOpt = opt.minimize(self.siameseLoss, var_list=siaVarList)

        # Optimizer VAE
        vaeVarList = [x for x in tf.global_variables() if x.name.startswith('vae_')]
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.vaeOpt = opt.minimize(self.loss, var_list=vaeVarList)

    def fit(self, sess):
        print("Begin Session: Processing {:d} images in {:d} batches".format(
              self.batch_size * self.n_batches * self.epochs, self.n_batches))
        # Session
        sess.run(tf.global_variables_initializer())
        sess.run(self.iterator.initializer)
        self.err = False

        # train the siamese network
        for epoch in range(100):
            print("Epoch {:>2}:  ".format(epoch), end="")
            for i in range(self.n_batches - 2):
                print(".", end="")
                X, Y = self.get_next(sess)
                tloss, _ = sess.run([self.siameseLoss, self.siaOpt],
                                    feed_dict={self.input: X, self.siameseLabel: Y})
            print("\nLoss= {:>8.4f} Time: {:>8.4f}".format(tloss))

        # train the VAE
        for epoch in range(self.epochs):
            tic = time()
            try:
                print("Epoch {:>2}:  ".format(epoch), end="")
                for i in range(self.n_batches - 2):
                    print(".", end="")
                    X = self.get_next(sess)
                    tloss, _ = sess.run([self.loss, self.vaeOpt],
                                        feed_dict={self.input: X})
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
        return sess.run(self.model, feed_dict={self.input: x})

    def create_iterator(self):
        # Create Tensorflow Iterator
        label_ds = tf.data.Dataset.from_tensor_slices(self.all_labels)
        path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        full_ds = tf.data.Dataset.zip((path_ds, label_ds))

        ds = full_ds.map(self.load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        ds = ds.shuffle(buffer_size=self.batch_size * 2)
        ds = ds.repeat(self.data_repeats)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        self.iterator = ds.make_initializable_iterator()
        return self.iterator.get_next()

    def get_next(self, sess):
        return sess.run(tf.layers.Flatten()(self.next_element))


    def siamese(self, input):
        self.siameseInput = tf.placeholder_with_default(input, input.shape, name='siameseInput')

        convA = self.featExtractorLayer(input, "siamese_A")
        convB = self.featExtractorLayer(self.siameseInput, 'siamese_B')

        latentRepr = tf.concat([convA, convB], axis=0)
        return tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(latentRepr)


    def featExtractorLayer(self, input, scopeName):
        # 64 max pool 256 max pool flatten drop dense256 dense100 out
        with tf.variable_scope(scopeName, reuse=tf.AUTO_REUSE):
            model = tf.reshape(input, [-1, self.img_side, self.img_side, self.n_channels])
            model = tf.keras.layers.Conv2D(8, kernel_size=3, activation=tf.nn.relu)(model)
            model = tf.keras.layers.MaxPool2D()(model)
            model = tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu)(model)
            model = tf.keras.layers.MaxPool2D()(model)
            model = tf.keras.layers.Flatten()(model)
        return model


    def encode(self, x):
        with tf.variable_scope("vae_encoder", reuse=tf.AUTO_REUSE):
            model = tf.reshape(x, [-1, self.img_side, self.img_side, self.n_channels])
            model = tf.layers.Conv2D(filters=6 , kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(model)
            model = tf.layers.Conv2D(filters=12, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(model)
            model = tf.layers.Conv2D(filters=18, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(model)

            model = tf.layers.Flatten()(model)
            self.mu = tf.layers.Dense(self.n_z)(model)
            self.log_sigma_sq = tf.layers.Dense(self.n_z)(model)
            self.eps = tf.random_normal(shape=tf.shape(self.log_sigma_sq),
                                        mean=0, stddev=1, dtype=tf.float32)
            self.z = self.mu + tf.sqrt(tf.exp(self.log_sigma_sq)) * self.eps
            # return z, log_sigma_sq, mu

    def decode(self, x):
        with tf.variable_scope("vae_decoder", reuse=tf.AUTO_REUSE):
            model = tf.layers.Dense(7*7*32)(x)
            model = tf.reshape(model, [-1, 7, 7, 32])
            model = tf.layers.Conv2DTranspose(filters=18, kernel_size=3, strides=(2, 2), padding="same", activation=tf.nn.relu)(model)
            model = tf.layers.Conv2DTranspose(filters=12, kernel_size=3, strides=(2, 2), padding="same", activation=tf.nn.relu)(model)
            model = tf.layers.Conv2DTranspose(filters=6, kernel_size=3, strides=(2, 2), padding="same", activation=tf.nn.relu)(model)
            model = tf.layers.Flatten()(model)
            model = tf.layers.Dense(self.img_size, activation=tf.nn.sigmoid)(model)
        return model

    def load_data(self, path, format="jpg", extra_repeats=2):
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob("*." + format))
        self.all_image_paths = [str(path) for path in all_image_paths]
        self.all_labels = [str(l)[len(str(data_root))+1:].split('_')[0] for l in all_image_paths]
        #random.shuffle(self.all_image_paths)
        self.image_count = len(self.all_image_paths)
        self.data_repeats = self.epochs * extra_repeats
        self.n_batches = self.image_count // self.batch_size

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=self.n_channels)
        # image = tf.image.resize_images(image, [self.img_side, self.img_side])
        # image = tf.image.per_image_standardization(image)
        # image = image + tf.reduce_min(image)
        image /= tf.reduce_max(image)
        return image

    def load_and_preprocess_image(self, path, label):
        image = tf.read_file(path)
        return self.preprocess_image(image), label
