from BoostedModel import BoostedBetaVAE
import tensorflow as tf
import CKLoader as CKLoader

# call this function once to unzip the dataset
x,y = CKLoader.load()

model = BoostedBetaVAE()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.fit(sess)

#with tf.Session() as sess:
#    model.fit(sess, "./ck")
