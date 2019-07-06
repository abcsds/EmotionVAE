from Model import BetaVAE
import tensorflow as tf
import CKLoader as CKLoader

# call this function once to unzip the dataset
CKLoader.unzip()

model = BetaVAE()

#with tf.Session() as sess:
#    model.fit(sess, "./ck")
