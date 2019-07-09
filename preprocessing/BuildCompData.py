###########################################################
# This is meant to run offline in my Laptop, not in Colab #
###########################################################

import cv2
import os
import shutil
import numpy as np
import preprocessing.CvUtil as CvUtil
import matplotlib.pyplot as plt

srcDir = os.path.realpath('../..') + '/CK/augmented/'
outDir = os.path.realpath('../..') + '/CK/comp/'

# clean output folder
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.mkdir(outDir)
os.mkdir(os.path.join(outDir, 'A'))
os.mkdir(os.path.join(outDir, 'B'))

allImages = os.listdir(srcDir)

def buildAvrgImg(filteredImages, numImg=70):
    np.random.shuffle(filteredImages)
    filteredImages = filteredImages[:numImg]

    avrg = np.zeros([128, 128])
    for imgFile in filteredImages:
        img = cv2.imread(os.path.join(srcDir, imgFile))[:,:,0]
        avrg += img

    return avrg / len(filteredImages)

imgID = 1

def createPositiveSamples(filteredImages, fileList, numIterations=30):
    global imgID

    for i in range(numIterations):
        imgA = cv2.imread(os.path.join(srcDir, filteredImages[np.random.randint(len(filteredImages))]))[:,:,0]
        imgB = cv2.imread(os.path.join(srcDir, filteredImages[np.random.randint(len(filteredImages))]))[:,:,0]

        # single-single comparison
        imgFile = '1_'+str(imgID)+'_ii.jpg'
        cv2.imwrite(os.path.join(outDir, 'A', imgFile), imgA)
        cv2.imwrite(os.path.join(outDir, 'B', imgFile), imgB)
        fileList.append(imgFile)
        imgID += 1

        # single-avrg comparison
        imgA = cv2.imread(os.path.join(srcDir, filteredImages[np.random.randint(len(filteredImages))]))[:,:,0]
        avrg = buildAvrgImg(filteredImages)
        imgFile = '1_'+str(imgID)+'_ia.jpg'
        cv2.imwrite(os.path.join(outDir, 'A', imgFile), imgA)
        cv2.imwrite(os.path.join(outDir, 'B', imgFile), avrg)
        fileList.append(imgFile)
        imgID += 1

        # avrg-avrg comparison
        avrgA = buildAvrgImg(filteredImages, numImg=20)
        avrgB = buildAvrgImg(filteredImages, numImg=20)
        imgFile = '1_'+str(imgID)+'_aa.jpg'
        cv2.imwrite(os.path.join(outDir, 'A', imgFile), avrgA)
        cv2.imwrite(os.path.join(outDir, 'B', imgFile), avrgB)
        fileList.append(imgFile)
        imgID += 1


def createNegativeSamples(emotion, filesByEmotion, fileList, numIterations=30):
    global imgID

    targetFiles = filesByEmotion[emotion]
    otherFiles = []
    for e in [e for e in range(7) if e is not emotion]:
        otherFiles += filesByEmotion[e]

    for i in range(numIterations):
        # single-single comparison
        imgA = cv2.imread(os.path.join(srcDir, targetFiles[np.random.randint(len(targetFiles))]))[:,:,0]
        imgB = cv2.imread(os.path.join(srcDir, otherFiles[np.random.randint(len(otherFiles))]))[:,:,0]
        imgFile = '0_'+str(imgID)+'_ii.jpg'
        cv2.imwrite(os.path.join(outDir, 'A', imgFile), imgA)
        cv2.imwrite(os.path.join(outDir, 'B', imgFile), imgB)
        fileList.append(imgFile)
        imgID += 1

        # single-avrg comparison
        imgA = cv2.imread(os.path.join(srcDir, targetFiles[np.random.randint(len(targetFiles))]))[:,:,0]
        otherEmotion = emotion
        while otherEmotion == emotion:
            otherEmotion = np.random.randint(7)
        avrg = buildAvrgImg(filesByEmotion[otherEmotion])
        imgFile = '0_'+str(imgID)+'_ia.jpg'
        cv2.imwrite(os.path.join(outDir, 'A', imgFile), imgA)
        cv2.imwrite(os.path.join(outDir, 'B', imgFile), avrg)
        fileList.append(imgFile)
        imgID += 1

        # avrg-avrg comparison
        avrgA = buildAvrgImg(filesByEmotion[emotion], numImg=20)
        otherEmotion = emotion
        while otherEmotion == emotion:
            otherEmotion = np.random.randint(7)
        avrgB = buildAvrgImg(filesByEmotion[otherEmotion], numImg=20)
        imgFile = '0_'+str(imgID)+'_aa.jpg'
        cv2.imwrite(os.path.join(outDir, 'A', imgFile), avrgA)
        cv2.imwrite(os.path.join(outDir, 'B', imgFile), avrgB)
        fileList.append(imgFile)
        imgID += 1



fileList = []
filesByEmotion = {}

for emotion in range(7):
    filteredImages = [i for i in allImages if i.startswith(str(emotion))]
    filesByEmotion[emotion] = filteredImages

for emotion in range(7):
    filteredImages = filesByEmotion[emotion]
    createPositiveSamples(filteredImages, fileList)
    createNegativeSamples(emotion, filesByEmotion, fileList)

with open(os.path.join(outDir, 'fileList.txt'), 'w') as file:
    for line in fileList:
        file.write(line+'\n')

zipFileName = 'comp.zip'
if os.path.exists(zipFileName):
    os.remove(zipFileName)
import zipfile
with zipfile.ZipFile(zipFileName, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(os.path.join(outDir, 'fileList.txt'), arcname='fileList.txt')
    for file in os.listdir(os.path.join(outDir, 'A')):
        zipf.write(os.path.join(outDir, 'A', file), arcname='A/'+file)
    for file in os.listdir(os.path.join(outDir, 'B')):
        zipf.write(os.path.join(outDir, 'B', file), arcname='B/'+file)