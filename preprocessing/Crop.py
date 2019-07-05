import cv2
import os
import shutil

###########################################################
# This is meant to run offline in my Laptop, not in Colab #
###########################################################

ckDir = os.path.realpath('../..') + '/CK/CK/cohn-kanade'
outDir = os.path.realpath('../..') + '/CK/cropped/'

# use the last N images
USED_IMAGES = 3
imgNumber = 0
TARGET_DIM = 128

# face detection model
face_cascade = cv2.CascadeClassifier()
success = face_cascade.load('haarcascade_frontalface_default.xml')

# clean output folder
if os.path.exists(outDir):
    shutil.rmtree(outDir)
    os.mkdir(outDir)

maxEmo = -1

for actor in os.listdir(ckDir):
    actorDir = os.path.join(ckDir, actor)

    for emotion in os.listdir(actorDir):
        maxEmo = max(maxEmo, int(emotion))
        if int(emotion) <= 6:
            imgDir = os.path.join(actorDir, emotion)

            # take only the first and last n images
            images = sorted(os.listdir(imgDir))
            images = images[-USED_IMAGES:]

            for imgFile in images:
                img = cv2.imread(os.path.join(imgDir, imgFile))
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                except: pass # image is already grey

                faces = face_cascade.detectMultiScale(img)
                if len(faces) != 1:
                    print("unsuspected number of faces:", len(faces), 'file:', actor, emotion, imgFile)#
                    continue

                imgNumber += 1
                outFile = emotion[2] + '_' + str(imgNumber) + '.jpg'

                # crop to the face
                x, y, w, h = faces[0]
                cropped = img[y:y+h, x:x+w]

                cropped = cv2.resize(cropped, (TARGET_DIM, TARGET_DIM))
                cv2.imwrite(os.path.join(outDir, outFile), cropped)

print(maxEmo)