import tensorflow as tf
# from BatchIterator import BatchIterator # Batch iterator is tensorflow-based
import pathlib
import numpy as np
from time import time
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfd = tfp.distributions


# think about
# session management (session should come from colab)
# is the optimizer included in the model class? (sklearn style)
# logging for tensorboard (summaries)
class BetaVAE:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    def __init__(self,
                 path="./ck",
                 img_side=128,
                 batch_size=16,
                 epochs=10,
                 n_z=6,
                 beta=1,
                 gamma=1,
                 lr=1e-3,
                 format="jpg",
                 extra_repeats=2,
                 n_channels=1,
                 logdir="./graph"):
        self.img_side   = img_side
        self.img_size   = self.img_side**2
        self.batch_size = batch_size
        self.epochs     = epochs
        self.n_z        = n_z
        self.beta       = beta
        self.gamma      = gamma
        self.lr         = lr
        self.n_channels = n_channels
        self.format     = format
        self.n_classes  = self.n_z
        self.logdir     = logdir


        # Build Model Architecture
        tf.reset_default_graph()

        # Data loading
        self.load_data(path, extra_repeats=extra_repeats)
        self.next_element = self.create_iterator()

        # Placeholders
        self.input  = tf.placeholder(tf.float32,
                                     [None, self.img_size],
                                     name="Inputs")
        self.latent = tf.placeholder(tf.float32,
                                     [None, self.n_z],
                                     name="Latents")
        self.label  = tf.placeholder(tf.float32,
                                     [None, self.n_classes],
                                     name="Labels")

        # Encoder, decoder and sampling
        self.z, self.theta, self.k = self.encode(self.input)
        self.model  = self.decode(self.z)
        self.sample = self.decode(self.latent)

        # LOSSES
        # Reconstruction loss
        # Minimize the binary cross-entropy loss
        self.recon_loss = tf.keras.backend.binary_crossentropy(self.input, self.model)
        self.recon_loss = tf.reduce_mean(self.recon_loss,
                                         name="Reconstruction_Loss")

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between
        # the latent distribution and Gamma(0, 1)
        dist1 = tfd.Gamma(self.k, self.theta)
        dist2 = tfd.Gamma(1, 1)
        self.latent_loss = tfd.kl_divergence(dist1, dist2)
        self.latent_loss = tf.reduce_mean(self.latent_loss, name="Latent_Loss")

        # Classification Loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label,
                                                          logits=self.z)
        self.class_loss = tf.reduce_mean(loss, name="Classification_Loss")

        # Total loss
        self.loss  = self.recon_loss
        self.loss += self.beta  * self.latent_loss
        self.loss += self.gamma * self.class_loss
        self.plottableLoss = []

        # Optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.opt = opt.minimize(self.loss)

    def fit(self, sess, viz=False):
        print("Begin Session: Processing {:d} images in {:d} batches".format(
              self.batch_size * self.n_batches * self.epochs, self.n_batches))
        # Session
        sess.run(tf.global_variables_initializer())
        sess.run(self.iterator.initializer)
        writer = tf.summary.FileWriter(self.logdir, sess.graph)
        zs       = []
        ys       = []
        for epoch in range(self.epochs):
            tic = time()
            try:
                print("Epoch {:>2}:  ".format(epoch), end="")
                for i in range(self.n_batches):
                    print(".", end="")
                    X, y = self.get_next(sess)
                    # One hot encode y
                    feed_dict = {self.input: X,
                                 self.label: self.oneHotEncode(y)}
                    tloss, _  = sess.run([self.loss, self.opt],
                                         feed_dict=feed_dict)
                    if np.isnan(tloss) or np.isinf(tloss):
                        raise ValueError("Loss went cray cray")
                    self.plottableLoss.append(tloss)
                    if epoch == self.epochs-1:
                        feed_dict = {self.input: X,
                                     self.label: self.oneHotEncode(y)}
                        lat = sess.run(self.z, feed_dict=feed_dict)
                        zs.append(lat)
                        ys.append(y)
                print("\nLoss= {:>8.4f} Time: {:>8.4f}".format(tloss,
                                                               time() - tic))
            except tf.errors.OutOfRangeError:
                pass
            if viz:
                self.plot_iteration(sess, X, y)
        self._lat_reps = np.vstack(zs)
        self._labels   = np.vstack(ys)
        if viz:
            self.plot_interactions(sess, X)
            self.plot_independent(sess, X, y)
        writer.close()

    def predict(self, sess, X, y):
        feed_dict = {self.input: X,
                     self.label: self.oneHotEncode(y)}
        return sess.run(self.model, feed_dict=feed_dict)

    def create_iterator(self):
        # Create Tensorflow Iterator
        path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.all_image_labels, tf.int64))
        image_ds = path_ds.map(self.load_and_preprocess_image,
                               num_parallel_calls=self.AUTOTUNE)
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        ds = ds.shuffle(buffer_size=self.batch_size * 2)
        ds = ds.repeat(self.data_repeats)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        self.iterator = ds.make_initializable_iterator()
        return self.iterator.get_next()

    def get_next(self, sess):
        X, y = sess.run(self.next_element)
        # Don't create graph on the run
        X    = X.reshape(-1, self.img_size)
        y = y.reshape(-1, 1)
        return X, y

    def encode(self, x):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            model = tf.reshape(x, [-1, self.img_side, self.img_side, self.n_channels])
            model = tf.keras.layers.Conv2D(filters=32,  kernel_size=3, strides=(2,2), padding="same", activation="relu")(model)
            model = tf.keras.layers.Conv2D(filters=64,  kernel_size=3, strides=(2,2), padding="same", activation="relu")(model)
            self._int_shape = tf.keras.backend.int_shape(model)
            model = tf.keras.layers.Flatten()(model)
            k     = tf.keras.layers.Dense(self.n_z)(model)
            theta = tf.keras.layers.Dense(self.n_z)(model)
            eps   = tf.random.gamma(shape=tf.shape(theta),
                                    alpha=1,
                                    beta=1,
                                    name="GammaSample",
                                    dtype=tf.float32)
            z = k + theta * eps
            return z, theta, k

    def decode(self, x):
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            model = tf.keras.layers.Dense(self._int_shape[1] * self._int_shape[2] * self._int_shape[3])(x)
            model = tf.reshape(model, [-1, self._int_shape[1], self._int_shape[2], self._int_shape[3]])
            model = tf.keras.layers.Conv2DTranspose(filters=64,  kernel_size=3, strides=(2,2), padding="same", activation="relu")(model)
            model = tf.keras.layers.Conv2DTranspose(filters=32,  kernel_size=3, strides=(2,2), padding="same", activation="relu")(model)
            model = tf.keras.layers.Conv2DTranspose(filters=1,   kernel_size=1, padding="same", activation="relu")(model)
            model = tf.keras.layers.Flatten()(model)
            model = tf.keras.layers.Dense(self.img_size, activation="sigmoid")(model)
            return model

    def load_data(self, path, extra_repeats=2):
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob("*." + self.format))
        self.all_image_labels = [int(file.name[0]) for file in all_image_paths]
        self.all_image_paths = [str(path) for path in all_image_paths]
        self.image_count = len(self.all_image_paths)
        self.data_repeats = self.epochs * extra_repeats
        self.n_batches = self.image_count // self.batch_size

    def preprocess_image(self, image):
        if self.format in ["jpeg", "jpg"]:
            image = tf.image.decode_jpeg(image, channels=self.n_channels)
        elif self.format == "png":
            image = tf.image.decode_png(image, channels=self.n_channels)
        else:
            raise FileNotFoundError("Image format not accepted.")
        # image = tf.image.resize_images(image, [self.img_side, self.img_side])
        # image = tf.image.per_image_standardization(image)
        # image = image + tf.reduce_min(image)
        image /= tf.reduce_max(image)
        return image

    def load_and_preprocess_image(self, path):
        image = tf.read_file(path)
        return self.preprocess_image(image)

    def plot_iteration(self, sess, X, y, n_subplots=5, figsize=(10, 4)):
        # Generate images from noise
        f, a = plt.subplots(2, n_subplots, figsize=figsize)
        for i in range(n_subplots):
            inn = X
            inn = np.reshape(inn, newshape=(-1, self.img_side, self.img_side, 1))
            inn = np.reshape(np.repeat(inn[i][:, :, np.newaxis], 3, axis=2),
                             newshape=(self.img_side, self.img_side, 3))
            a[0][i].imshow(inn)
            a[0][i].axis("off")
            feed_dict = {self.input: X,
                         self.label: self.oneHotEncode(y)}
            out = sess.run(self.model, feed_dict=feed_dict)
            out = np.reshape(out, newshape=(-1, self.img_side, self.img_side, 1))
            out = np.reshape(np.repeat(out[i][:, :, np.newaxis], 3, axis=2),
                             newshape=(self.img_side, self.img_side, 3))
            a[1][i].imshow(out)
            a[1][i].axis("off")

        f.show()
        plt.show()

    def plot_independent(self, sess, X, y, n_subplots=5, figsize=(16, 16), scale=2):
        f, a = plt.subplots(self.n_z, n_subplots, figsize=figsize)
        samples = np.linspace(-scale, scale, n_subplots)
        for i in range(self.n_z):
            for j in range(n_subplots):
                r = np.array([[0] * self.n_z])
                r[0][i] = samples[j]
                feed_dict={self.input: X,
                           self.label: self.oneHotEncode(y),
                           self.latent: r}
                out = sess.run(self.sample, feed_dict=feed_dict)
                out = np.reshape(out, newshape=(self.img_side, self.img_side, 1))
                out = np.reshape(np.repeat(out[:, :, np.newaxis], 3, axis=2),
                                 newshape=(self.img_side, self.img_side, 3))
                a[i][j].imshow(out)
                a[i][j].axis("off")
        f.show()
        plt.show()

    def plot_interactions(self, sess, X, vars=(0, 1), n_subplots=10, figsize=(16, 16), scale=2):
        f, a = plt.subplots(n_subplots, n_subplots, figsize=figsize)
        samples = np.linspace(-scale, scale, n_subplots)
        for i in range(n_subplots):
            for j in range(n_subplots):
                r = np.array([[0] * self.n_classes])
                r[0][vars[0]] = samples[i]
                r[0][vars[1]] = samples[j]
                feed_dict={self.input: X,
                           self.latent: r}
                out = sess.run(self.sample, feed_dict=feed_dict)
                out = np.reshape(out, newshape=(self.img_side, self.img_side, 1))
                out = np.reshape(np.repeat(out[:, :, np.newaxis], 3, axis=2),
                                 newshape=(self.img_side, self.img_side, 3))
                a[i][j].imshow(out)
                a[i][j].axis("off")
        f.show()
        plt.show()

    def latent_std(self):
        return np.std(self._lat_reps, axis=0)

    def oneHotEncodeInt(self, x):
        x = x[0]
        vec = [0] * self.n_classes
        if x != 0:
            vec[x - 1]  = 1
        return vec

    def oneHotEncode(self, x):
        return np.apply_along_axis(self.oneHotEncodeInt, 1, x)
