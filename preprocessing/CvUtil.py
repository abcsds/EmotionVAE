import cv2
import numpy as np

def flipLeftRight(img):
    cv2.flip(img, 1, img)
    return img


def distort3D(img):
    # persepective transformation
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

    distortScale = 20
    distorts = np.random.randint(-distortScale, distortScale, (4, 2))

    ymax, xmax = img.shape[:2]

    originalPoints = np.float32([[0,0], [0,xmax], [ymax,0], [ymax, xmax]])
    newPoints = np.float32(originalPoints + distorts)

    outputSize = (img.shape[1], img.shape[0]) #newPoints[3] - newPoints[0]
    M = cv2.getPerspectiveTransform(originalPoints, newPoints)

    img = cv2.warpPerspective(img, M, outputSize)

    return img


def linearTrans(img):
    alpha = np.random.rand() * 1.5 + 1
    beta = np.random.randint(100)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


augmentationOptions = [
    flipLeftRight,
    distort3D,
    linearTrans
]

def randomAugmentation(img):
    global augmentationOptions
    i = np.random.randint(len(augmentationOptions))
    return augmentationOptions[i](img), augmentationOptions[i].__name__
