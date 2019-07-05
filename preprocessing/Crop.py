import cv2
import os
import shutil

###########################################################
# This is meant to run offline in my Laptop, not in Colab #
###########################################################

ckDir = os.path.realpath('../..') + '/CK/CK+/'
ckImgDir = 'cohn-kanade-images'
ckLabelDir = 'Emotion'

outDir = os.path.realpath('../..') + '/CK/cropped/'
outNeutralDir = os.path.realpath('../..') + '/CK/neutral/'

# use the last N images
USED_IMAGES = 3
TARGET_DIM = 128

# face detection model
face_cascade = cv2.CascadeClassifier()
success = face_cascade.load('haarcascade_frontalface_default.xml')

# clean output folders
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.mkdir(outDir)
if os.path.exists(outNeutralDir):
    shutil.rmtree(outNeutralDir)
os.mkdir(outNeutralDir)


def saveNeutral(path, outFile):
    img = cv2.imread(path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass  # image is already grey

    faces = face_cascade.detectMultiScale(img)
    if len(faces) != 1:
        return
    # crop to the face
    x, y, w, h = faces[0]
    cropped = img[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, (TARGET_DIM, TARGET_DIM))
    cv2.imwrite(os.path.join(outNeutralDir, outFile), cropped)


def saveEmotion(path, outFile):
    img = cv2.imread(path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass  # image is already grey

    faces = face_cascade.detectMultiScale(img)
    if len(faces) != 1:
        print("unsuspected number of faces:", len(faces), 'file:', outFile)
        return False

    # crop to the face
    x, y, w, h = faces[0]
    cropped = img[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, (TARGET_DIM, TARGET_DIM))
    cv2.imwrite(os.path.join(outDir, outFile), cropped)
    return True

actorList = os.listdir(os.path.join(ckDir, ckLabelDir))
for a in range(len(actorList)):
    print('#####', 'Actor', a, 'of', len(actorList), '#############')
    actor = actorList[a]
    actorDir = os.path.join(ckDir, ckLabelDir, actor)

    for session in os.listdir(actorDir):
        sessionDir = os.path.join(actorDir, session)
        labelFiles = os.listdir(sessionDir)

        if len(labelFiles) > 0:
            with open(os.path.join(sessionDir, labelFiles[0])) as f:
                line = f.readline()
            line = line.strip()
            emotion = int(line[0])

            if emotion <= 6:
                imgDir = os.path.join(ckDir, ckImgDir, actor, session)
                images = sorted(os.listdir(imgDir))

                # save two neutral faces per person
                for i in range(4):
                    outFile = '0_' + actor + '_' + str(i) + '.jpg'
                    saveNeutral(os.path.join(imgDir, images[i]), outFile)

                # save n faces with emotion
                imgNumber = 0
                i = 1
                while imgNumber < USED_IMAGES and i < len(images):
                    outFile = str(emotion) + '_' + actor + '_' + str(imgNumber) + '.jpg'
                    if saveEmotion(os.path.join(imgDir, images[-i]), outFile):
                        imgNumber += 1
                    i += 1

            else:
                print("unkown emotion", emotion, 'in', sessionDir)
