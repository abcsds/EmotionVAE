import tensorflow as tf
from BatchIterator import BatchIterator

# think about
# session management (session should come from colab)
# is the optimizer included in the model class? (sklearn style)
# logging for tensorboard (summaries)
class BetaVAE:

    def __init__(self):
        self.input = None # tf.placeholder...
        self.labels = None

        self.output = None

        self.loss = None # output - labels
        self.plottableLoss = None # if needed

        self.opt = None # Adam.minimize(self.loss)


    def fit(self, session, x, y):
        batches = BatchIterator(x, y, 100)
        for mbX, mbY in batches:
            pass


    def predict(self, session, x):
        # sesssion.run(self.output)
        pass
