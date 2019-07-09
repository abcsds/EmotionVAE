import os
import cv2
import numpy as np
import shutil

outDir = "compDataset"

def load(labelFormat="oneHot"):
    '''
    :param labelFormat: oneHot or sparse
    :return: x, y as numpy arrays
    '''
    global outDir

    unzip()

    # load file names
    with open(os.path.join(outDir, "fileList.txt")) as f:
        files = f.readlines()
    files = [f[:-1] for f in files] # remove trailing '\n'

    x = list()
    y = list()

    for imgFile in files:
        imgA = cv2.imread(os.path.join(outDir, 'A', imgFile))[:,:,0]
        imgB = cv2.imread(os.path.join(outDir, 'B', imgFile))[:,:,0]
        x.append(np.stack([imgA, imgB]))

        label = int(imgFile.split('_')[0])
        y.append(label)

    return np.array(x), np.array(y)
    #raise NotImplementedError("This function is not used at the moment")


def unzip():
    global outDir
    zipFile = "preprocessing/comp.zip"

    assert os.path.exists(zipFile)

    if not os.path.exists(outDir):
        os.mkdir(outDir)

    if len(os.listdir(outDir)) > 0:
        print('dataset folder is not empty... Not unpacking dataset')
        return

    shutil.unpack_archive(zipFile, outDir, 'zip')
    #import zipfile
    #zip_ref = zipfile.ZipFile(zipFile, 'r')
    #zip_ref.extractall(outDir)
    #zip_ref.close()