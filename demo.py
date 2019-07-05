from Model import BetaVAE
import tensorflow as tf


model = BetaVAE()

with tf.Session() as sess:
    model.fit(sess, "./ck")
