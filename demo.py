from BoostedModel import BoostedBetaVAE
import tensorflow as tf
import CKLoader as CK
import CompDataLoader as CD
import numpy as np

# call this function once to unzip the dataset
#x,y = CKLoader.load()
x, y = CD.load()
CK.unzip()

model = BoostedBetaVAE()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.prefit(sess, x, y)
    for _ in range(5):
        idx = np.random.randint(len(x))
        x1, x2, y1 = x[idx, 0, :, :].reshape(1, -1), \
                    x[idx, 1, :, :].reshape(1, -1), \
                    y[idx].reshape(1, -1)
        score = sess.run(model.compScore,feed_dict={model.input: x1,
                                             model.siameseInput: x2})
        print('label', y1, 'pred', score)


#with tf.Session() as sess:
#    model.fit(sess, "./ck")
