###########################################################
# This is meant to run offline in my Laptop, not in Colab #
###########################################################

import cv2
import os
import shutil
import numpy as np
import preprocessing.CvUtil as CvUtil

srcDir = os.path.realpath('../..') + '/CK/cropped/'
srcNeutralDir = os.path.realpath('../..') + '/CK/neutral/'
outDir = os.path.realpath('../..') + '/CK/augmented/'

# clean output folder
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.mkdir(outDir)

allFiles = os.listdir(srcDir)
filesByLabel = {}
minLabelCount = 100000
for l in range(1, 7):
    images = [f for f in allFiles if f.startswith(str(l))]
    minLabelCount = min(minLabelCount, len(images))
    filesByLabel[l] = images

destinationCount = 210
print('destinationCount', destinationCount)

# augment emotion images
for label, files in filesByLabel.items():
    print('label', label, '# src images', len(files), '# generated images', destinationCount - len(files))
    # augment images
    for i in range(destinationCount - len(files)):
        file = files[np.random.randint(len(files))]
        img = cv2.imread(os.path.join(srcDir, file))

        img, transCode = CvUtil.randomAugmentation(np.copy(img))

        fileName, extension = file.split('.')
        newFileName = fileName+"_"+str(i)+'_'+transCode+'.'+extension
        cv2.imwrite(os.path.join(outDir, newFileName), img)

# copy original faces
for file in allFiles:
    shutil.copyfile(os.path.join(srcDir, file),
                    os.path.join(outDir, file))

allFiles = os.listdir(srcNeutralDir)
np.random.shuffle(allFiles)
for file in allFiles[:destinationCount]:
    shutil.copyfile(os.path.join(srcNeutralDir, file),
                    os.path.join(outDir, file))

zipFileName = 'augmented.zip'
if os.path.exists(zipFileName):
    os.remove(zipFileName)
import zipfile
with zipfile.ZipFile(zipFileName, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in os.listdir(outDir):
        zipf.write(os.path.join(outDir, file), arcname=file)