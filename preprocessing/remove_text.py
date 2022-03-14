import os
import numpy as np
import cv2

from distutils.archive_util import make_archive
from imutils.object_detection import non_max_suppression
from multiprocessing import Pool
from tqdm.auto import tqdm

"""
src: https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
"""
source_dir = '../data/hateful_memes/img'
target_dir_masked = '../data/hateful_memes_masked'
target_dir_inpainted = '../data/hateful_memes_inpainted'
east_path = 'models/frozen_east_text_detection.pb'
width = 320
height = 320
min_confidence = 0.5
img_fns = [x for x in os.listdir(source_dir) if x.endswith('.png')]

def transform_image(img_fp):

    # load the input image and grab the image dimensions
    image = cv2.imread(img_fp)
    masked, inpainted = image.copy(), image.copy()
    (H, W) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    mask_for_inpainting = np.zeros(inpainted.shape[:2], np.uint8)
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(masked, (startX, startY), (endX, endY), (127, 127, 127), -1)
        cv2.rectangle(mask_for_inpainting, (startX, startY), (endX, endY), 255, -1)

    inpainted = cv2.inpaint(inpainted, mask_for_inpainting, 7, cv2.INPAINT_NS)

    return masked, inpainted


layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"
    ]
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(east_path)

def transform_and_save_image(img_fn):
    img_fp = os.path.join(source_dir, img_fn)
    img_fp_masked = os.path.join(target_dir_masked, img_fn)
    img_fp_inpainted = os.path.join(target_dir_inpainted, img_fn)

    img_masked, img_inpainted = transform_image(img_fp)
    cv2.imwrite(img_fp_masked, img_masked)
    cv2.imwrite(img_fp_inpainted, img_inpainted)

with Pool(64) as pool:
    #pool.map(transform_and_save_image, img_fns)
    for _ in tqdm(pool.imap_unordered(transform_and_save_image, img_fns), total=len(img_fns)):
        pass

# for img_fn in tqdm(img_fns):
#     img_fp = os.path.join(source_dir, img_fn)
#     img_fp_masked = os.path.join(target_dir_masked, img_fn)
#     img_fp_inpainted = os.path.join(target_dir_inpainted, img_fn)

#     img_masked, img_inpainted = transform_image(img_fp)
#     cv2.imwrite(img_fp_masked, img_masked)
#     cv2.imwrite(img_fp_inpainted, img_inpainted)