import numpy as np

# Wrapper class for the batching mechanism
class BatchIterator:

    #######################################
    # Usage:
    # batches = BatchIterator(xData, yData)
    # for mbX, mbY in batches:
    #    doStuff
    #######################################

    def __init__(self, x, y, batchSize):
        self.batchSize = int(batchSize)
        self.x = x
        self.y = y
        self.reset()


    def reset(self):
        r = np.arange(len(self.x))
        np.random.shuffle(r)
        self.batchOrder = r
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(len(self.x) / self.batchSize)

    def __next__(self):
        if self.i + self.batchSize >= len(self.x):
            self.reset()  # automatically reshuffles the data for next epoch
            raise StopIteration

        mbX = self.x[self.batchOrder[self.i:self.i + self.batchSize]]
        mbY = self.y[self.batchOrder[self.i:self.i + self.batchSize]]
        self.i += self.batchSize
        return mbX, mbY