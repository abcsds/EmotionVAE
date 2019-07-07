import os
import cv2
import numpy as np

def load(dir='dataset', labelFormat="oneHot"):
    '''
    :param dir: dataset directory
    :param labelFormat: oneHot or sparse
    :return: x, y as numpy arrays
    '''
    unzip()

    x = list()
    y = list()

    for imgFile in os.listdir(dir):
        cvImg = cv2.imread(os.path.join(dir, imgFile))
        x.append(cvImg[:,:,0])

        labelVec = np.zeros(7)
        label = int(imgFile.split('_')[0])
        labelVec[label] = 1
        y.append(labelVec)

    return np.array(x), np.array(y)
    #raise NotImplementedError("This function is not used at the moment")


def unzip():
    zipFile = "preprocessing/augmented.zip"
    outDir = "dataset"

    assert os.path.exists(zipFile)

    if not os.path.exists(outDir):
        os.mkdir(outDir)

    if len(os.listdir(outDir)) > 0:
        print('dataset folder is not empty... Not unpacking dataset')
        return

    import zipfile
    zip_ref = zipfile.ZipFile(zipFile, 'r')
    zip_ref.extractall(outDir)
    zip_ref.close()

#x,y = load()
#print(x.shape, y.shape)