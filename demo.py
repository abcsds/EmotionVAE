from BoostedModel import BoostedBetaVAE
import tensorflow as tf
import CKLoader as CK
import CompDataLoader as CD
import numpy as np

if __name__ == '__main__':
    x, y = CD.load()
    CK.unzip()

    ids = list(range(len(x)))
    np.random.shuffle(ids)
    splitPoint = int(len(ids) * 0.1)
    trainIds, testIds = ids[splitPoint:], ids[:splitPoint]

    xTrain, yTrain = x[trainIds], y[trainIds]
    xTest, yTest = x[testIds], y[testIds]

    model = BoostedBetaVAE(share_weights=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.prefit(sess, xTrain, yTrain, epochs=60)

        missclassified = 0
        print('known data')
        for _ in range(20):
            idx = np.random.randint(len(xTrain))
            x1, x2, y1 = xTrain[idx, 0, :, :].reshape(1, -1), \
                         xTrain[idx, 1, :, :].reshape(1, -1), \
                         yTrain[idx].reshape(1, -1)
            score = sess.run(model.compScore, feed_dict={model.input: x1,
                                                         model.siameseInput: x2})
            print('label', y1[0, 0], 'pred', np.round(score[0, 0], 0), score[0, 0])
            if y1[0, 0] != np.round(score[0, 0], 0): missclassified += 1
        print('missclassified', missclassified)

        missclassified = 0
        print('unknown data')
        for i in range(len(xTest)):
            x1, x2, y1 = xTest[i, 0, :, :].reshape(1, -1), \
                         xTest[i, 1, :, :].reshape(1, -1), \
                         yTest[i].reshape(1, -1)
            score = sess.run(model.compScore, feed_dict={model.input: x1,
                                                         model.siameseInput: x2})
            print('label', y1[0, 0], 'pred', np.round(score[0, 0], 0), score[0, 0])
            if y1[0, 0] != np.round(score[0, 0], 0): missclassified += 1
        print('missclassified', missclassified)
