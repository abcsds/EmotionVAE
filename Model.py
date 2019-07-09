import tensorflow as tf
# from BatchIterator import BatchIterator # Batch iterator is tensorflow-based
import pathlib
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


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
                 beta=4,
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
        self.format     = format
        self.n_classes  = self.n_z

        # Build Model Architecture
        tf.reset_default_graph()

        # Data loading
        self.load_data(path, extra_repeats=extra_repeats)
        self.next_element = self.create_iterator()

        # OneHotEncoder
        self.enc = OneHotEncoder(handle_unknown="ignore", n_values=self.n_classes)
        self.enc.fit(np.arange(self.n_classes).reshape(-1, 1))

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
        self.encode(self.input) # z, mu, and lss are saved in the object.
        self.model = self.decode(self.z)

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
        self.recon_loss = tf.reduce_mean(self.recon_loss,
                                         name="Reconstruction_Loss")

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between
        # the latent distribution and N(0, 1)
        self.latent_loss = -0.5 * tf.reduce_sum(
                            1 + self.log_sigma_sq - tf.square(self.mu) -
                            tf.exp(self.log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(self.latent_loss, name="Latent_Loss")

        # Classification Loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label,
                                                          logits=self.z)
        self.class_loss = tf.reduce_mean(loss, name="Classification_Loss")

        # Total loss
        self.loss = self.recon_loss + self.beta * self.latent_loss + self.class_loss
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
        self.err = False
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
                                 self.label: self.enc.transform(y).toarray()}
                    tloss, _  = sess.run([self.loss, self.opt],
                                         feed_dict=feed_dict)
                    if np.isnan(tloss) or np.isinf(tloss):
                        print("Error: loss went cray cray")
                        self.err = True
                        break
                    self.plottableLoss.append(tloss)
                    if epoch == self.epochs-1:
                        feed_dict = {self.input: X,
                                     self.label: self.enc.transform(y).toarray()}
                        lat = sess.run(self.z, feed_dict=feed_dict)
                        zs.append(lat)
                        ys.append(y)
                if self.err:
                    break
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
            self.plot_independant(sess, X, y)

    def predict(self, sess, X, y):
        feed_dict = {self.input: X,
                     self.label: self.enc.transform(y).toarray()}
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
        X    = sess.run(tf.keras.layers.Flatten()(X))
        y = y.reshape(-1, 1)
        return X, y

    def encode(self, x):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            model = tf.reshape(x, [-1, self.img_side, self.img_side, self.n_channels])
            model = tf.keras.layers.Conv2D(filters=64,  kernel_size=5, padding="same", activation="relu")(model)
            model = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(model)
            model = tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding="same", activation="relu")(model)
            model = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(model)
            model = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding="same", activation="relu")(model)
            model = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(model)
            self._int_shape = tf.keras.backend.int_shape(model)
            model = tf.keras.layers.Flatten()(model)
            self.mu = tf.keras.layers.Dense(self.n_z)(model)
            self.log_sigma_sq = tf.keras.layers.Dense(self.n_z)(model)
            self.eps = tf.random_normal(shape=tf.shape(self.log_sigma_sq),
                                        mean=0, stddev=1, dtype=tf.float32)
            self.z = self.mu + tf.sqrt(tf.exp(self.log_sigma_sq)) * self.eps
            # return z, log_sigma_sq, mu

    def decode(self, x):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            model = tf.keras.layers.Dense(self._int_shape[1] * self._int_shape[2] * self._int_shape[3])(x)
            model = tf.reshape(model, [-1, self._int_shape[1], self._int_shape[2], self._int_shape[3]])
            model = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, padding="same", activation="relu")(model)
            model = tf.keras.layers.UpSampling2D((2, 2))(model)
            model = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, padding="same", activation="relu")(model)
            model = tf.keras.layers.UpSampling2D((2, 2))(model)
            model = tf.keras.layers.Conv2DTranspose(filters=64,  kernel_size=5, padding="same", activation="relu")(model)
            model = tf.keras.layers.UpSampling2D((2, 2))(model)
            model = tf.keras.layers.Conv2DTranspose(filters=1,   kernel_size=1, padding="same", activation="sigmoid")(model)
            model = tf.keras.layers.Flatten()(model)
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
                         self.label: self.enc.transform(y).toarray()}
            out = sess.run(self.model, feed_dict=feed_dict)
            out = np.reshape(out, newshape=(-1, self.img_side, self.img_side, 1))
            out = np.reshape(np.repeat(out[i][:, :, np.newaxis], 3, axis=2),
                             newshape=(self.img_side, self.img_side, 3))
            a[1][i].imshow(out)
            a[1][i].axis("off")

        f.show()
        plt.show()

    def plot_independant(self, sess, X, y, n_subplots=5, figsize=(16, 16), scale=5):
        f, a = plt.subplots(self.n_z, n_subplots, figsize=figsize)
        samples = np.linspace(-scale, scale, n_subplots)
        for i in range(self.n_z):
            for j in range(n_subplots):
                r = np.array([[0] * self.n_z])
                r[0][i] = samples[j]
                out = sess.run(self.sample, feed_dict={self.latent: r})
                out = np.reshape(out, newshape=(self.img_side, self.img_side, 1))
                out = np.reshape(np.repeat(out[:, :, np.newaxis], 3, axis=2),
                                 newshape=(self.img_side, self.img_side, 3))
                a[i][j].imshow(out)
                a[i][j].axis("off")
        f.show()
        plt.show()

    def plot_interactions(self, sess, X, vars=(0, 1), n_subplots=10, figsize=(16, 16), scale=5):
        f, a = plt.subplots(n_subplots, n_subplots, figsize=figsize)
        samples = np.linspace(-scale, scale, n_subplots)
        for i in range(n_subplots):
            for j in range(n_subplots):
                r = np.array([[0] * self.n_classes])
                r[0][vars[0]] = samples[i]
                r[0][vars[1]] = samples[j]
                # Doesn't need a label, because there is no classification
                out = sess.run(self.sample, feed_dict={self.latent: r})
                out = np.reshape(out, newshape=(self.img_side, self.img_side, 1))
                out = np.reshape(np.repeat(out[:, :, np.newaxis], 3, axis=2),
                                 newshape=(self.img_side, self.img_side, 3))
                a[i][j].imshow(out)
                a[i][j].axis("off")
        f.show()
        plt.show()

    def latent_std(self):
        return np.std(self._lat_reps, axis=0)
