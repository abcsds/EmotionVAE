import numpy as np
import os
import cv2

def load(dir='dataset', labelFormat="oneHot"):
    '''
    :param dir: dataset directory
    :param labelFormat: oneHot or sparse
    :return: x, y as numpy arrays
    '''
    pass


def unzip():
    assert os.path.exists("augmented.zip") and os.path.exists("dataset")
    if len(os.listdir("dataset")) > 0:
        print('dataset folder is not empty... Not unpacking dataset')
        return

    import zipfile
    zip_ref = zipfile.ZipFile("augmented.zip", 'r')
    zip_ref.extractall("dataset/")
    zip_ref.close()

#x,y = load()
#print(x.shape, y.shape)