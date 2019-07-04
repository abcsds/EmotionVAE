import numpy as np
import os
import cv2

def load(dir='dataset', labelFormat="oneHot"):
    '''
    :param dir: dataset directory
    :param labelFormat: oneHot or sparse
    :return: x, y as numpy arrays
    '''
    x = list()
    y = list()

    for file in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, file))

        # remove the 3rd image dimension as all 3 just have the grey value
        img = img[:,:,0].reshape(img.shape[0], img.shape[0])

        x.append(img)

        sparseLabel = int(file.split('_')[0])
        if labelFormat == 'oneHot':
            hotVec = np.zeros(6)
            hotVec[sparseLabel-1] = 1
            y.append(hotVec)
        else:
            y.append(sparseLabel)

    return np.array(x), np.array(y)

#x,y = load()
#print(x.shape, y.shape)