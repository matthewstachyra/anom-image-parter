import os
import re
import cv2
import sys
sys.setrecursionlimit(10000)
import random
import pandas as pd
import numpy as np
from typing import List
from PIL import Image
from matplotlib import pyplot as plt


#####################################################
#                                                   #
#                                                   #
# Core image partioning algorithms                  #
#                                                   #
#                                                   #
#####################################################

def imPartition(path, images, reference, splices, blackthresh=0.80, bminpixel=5, anomthresh=0.10):
    """returns (dim)**2 dimension images as numpy arrays with a tag - 0 or 1, indicating whether an anomaly is
       present or not -- provided they do not exceed the exclusion threshold (i.e., are not too black).

    :param:
    :param:
    :param:
    :param:
    :returns:
    :rtype:
    """
    # NOTE: may need to flatten this as it could be a list of lists of tuples
    return [*map(lambda x : part(path, x, reference, splices, blackthresh, bminpixel, anomthresh),
               [tup[1] for tup in list(images.itertuples())])]  # list of image names/paths


def part(path, im, ref, splices, bthresh=0.8, minbpixel=5, athresh=0.1):
    """returns labeled (0 - nonanomalous, 1 - anomalous) partitions of the inputted image

    :param:
    :param:
    :param:
    :param:
    :returns:
    :rtype:

    NOTE  splices has form [[(rowindex1, rowindex2), (colindex1, colindex2), ...]]
    """
    # termination condition 1:
    # the image doesn't return a proper tag or contains only the anomaly
    tag = getTag(im)
    if not tag or "anomaly_only_view" in im: return

    # part input image according to input slices, keeping those partitions that are not too
    # black and labeling them according to whether they are anomalous or not
    refim = ref[tag]
    npim  = toNP(path, im)
    return [labelPart(im, npim, refim, s[0], s[1], athresh)  # s[0] is the row tuple and s[1] is the column tuple
                      for s in splices
                      if checkPart(npim, s[0], s[1], bthresh, minbpixel)]


def buildReference(path, ims, minpixel=5):
    """returns dictionary of tag:reference pairs where tag is the "P/d/d" anomaly tag and the reference
       is an array with all 0s except for 1s where an anomaly is present at that pixel

    :param:
    :param:
    :param:
    :param:
    :returns:
    :rtype:
    """
    # helper: builds reference image that "traces" the anomaly and returns
    # an array with 1s at pixels where anomaly is
    def build(reference):
        zeros = np.zeros((len(reference), len(reference[0])))
        for i in range(len(reference)):
            for j in range(len(reference[i])):
                if np.all(reference[i][j] > minpixel):
                    BFS(i, j, reference, zeros)
        return zeros

    # assumption: there is only one anomaly in each image
    # helper: identifies which pixels contain anomaly using breadth first search,
    # looping through every pixel until we find an anomalous one, then considering every
    # neighbor that is anomalous until none are left / all are visited
    def BFS(row, col, reference, zeros):
        if row > len(reference)-1 or col > len(reference[0])-1 or row <0 or col <0: return
        if zeros[row][col]==1: return

        pixel = reference[row][col]
        if np.all(pixel > minpixel):
            zeros[row][col] = 1
        else:
            return

        moves = [(1,1), (-1,-1), (1,-1), (-1,1), (1,0), (0,1), (-1,0), (0,-1)]
        for m in moves:
            BFS(row+m[0], col+m[1], reference, zeros)


    # assumption: there exists an "anomaly only view" version of every image that can be "traced" to
    # identify which pixels are anomalous versus not anomalous using the above build and BFS methods
    # design: for each tag, builds reference image from zeros array by running BFS on the first pixel
    # that is non black, converting 0s to 1s for all adjacent pixels that are non black
    references = {}
    for i in ims.iterrows():
        imname = i[1][0]
        tag = getTag(imname) if getTag(imname) else ""
        if tag and tag not in references.keys():
            func = lambda x: ("anomaly_only_view" in str(x)) and (tag in str(x))

            # grab the next image (as its name in the df) that is a full image and has the anomaly in
            # the correct position
            reference = next(filter(func, [tup[1]  # holds the image name, whereas tup[0] holds the index
                                           for tup
                                           in list(ims.itertuples())]))
            references[tag] = build(toNP(path, reference)[:, :, :3])
    return references


def createSplices(path, im, mode, dim, k=None):
    """returns list of splices according to which to partition the image to.

    :param:
    :param:
    :param:
    :param:
    :returns:
    :rtype:

    NOTE  can add new modes for splicing in the future easily here.
    """
    if mode == 'default' and len(toNP(path, im))%dim!=0:
        raise AttributeError("'dim' of %d does not evenly divide image dimension %d by %d" % (xdim, len(npim), len(npim[0])))
    if mode == 'default':
        return defaultSplice(path, im, dim)
    if mode == 'feature':
        return featureSplice(path, im, dim, k)


def defaultSplice(path, im, xydim):
    """return list of dim by dim splices for the images.

    :param:
    :param:
    :returns:
    :rtype:
    """
    npim = toNP(path, im)
    return [[(r, r+xydim),(c, c+xydim)]
            for r in range(0, len(npim), xydim)
            for c in range(0, len(npim), xydim)]

def featureSplice(path, im, ydim, k):
    """return list of 2-tuples with indices of partitions of image, where image is a numpy array,
    integer dim is used to generate the row windows of the partitions, and integer k is the number
    of features to extract

    :param:
    :param:
    :param:
    :returns:
    :rtype:

    NOTE  currently sets a uniform dimension for every image
    """
    # build kLargestDiffs to store differences between successive column sums of pixel values
    # assumption: large differences in column pixel sums represent edges (such as
    # the truck door hinges)
    # algorithm: get sum of pixel values for each column. then get difference between column
    # i and i-1. the greatest differences represent the greatest changes from dark to light,
    # or from no feature to some feature in the image.
    featureWidth = 5
    batchSize    = ydim // featureWidth
    npim         = toNP(path, im)
    summedImage  = npim[:,:,:3].sum(axis=2)  # drop alpha channel and sum RGB channels
    colSums      = [sum(summedImage[:, c]) for c in range(len(summedImage))]
    batchDiffs, maxDiffs, kMaxDiffs = [], [], []
    for i in range(1, len(colSums)):
        diff = abs((colSums[i] - colSums[i-1]))
        if len(kMaxDiffs) < k: kMaxDiffs.append((i, diff))  # 2-tuples as (index, difference)
        batchDiffs.append((i, diff))

        # assumptions: no vertical feature occurs more often than every 5 pixels and
        # the cargo is at least 5 pixels wide, so this consitutes the featureWidth value
        # algorithm: batch 10 pixel sums at a time to ensure we only get 1 column
        # per local feature.
        if i%featureWidth==0:
            maxBatchDiff = sorted(batchDiffs, key=lambda tup : tup[1], reverse=True)[0]
            maxDiffs.append(maxBatchDiff)
            batchDiffs = []

        # for each batch, grab the largest difference and append
        # to k largest diffs if its larger than any, replacing the smallest. this
        # per batch process also ensures an O(n) sort rather than O(nlogn) sort.
        if i%batchSize==0:
            maxDiff = sorted(maxDiffs, key=lambda tup : tup[1], reverse=True)[0][1]
            if any(list(map(
                    lambda x : x[1] < maxDiff,
                    kMaxDiffs))):
                diffOnly                       = [x[1] for x in kMaxDiffs]
                minLargest                     = min(diffOnly)
                minLargestIndex                = list(diffOnly).index(minLargest)
                kMaxDiffs[minLargestIndex]     = (i, maxDiff)
            maxDiffs = []

    # assumption: hinges in door are equidistant so we can set the new x to be the
    # dfference between the first two values
    # algorithm: create k+1 windows of new x dimension by old y dimension (the
    # parameter dim) using kLargestDiffs, which is a list of 2-tuples of form
    # (index, difference) with the k largest differences in the image.
    kMaxDiffs.sort(key=lambda x : x[0])  # sort by index
    xdim       = int(kMaxDiffs[1][0] - kMaxDiffs[0][0])  # set uniform x dimension
    startIndex = kMaxDiffs[0][0]  # grab first index
    return [[(r, r+ydim),(c, c+xdim)]
            for c in range(startIndex, xdim*(k), xdim)
            for r in range(0, len(npim), ydim)]


def checkPart(im, rsplice, csplice, bthresh, bminpixel):
    """returns True if partition is OK to include (i.e., is not too black); False, otherwise.

    :param:
    :param:
    :param:
    :param:
    :param:
    :returns:
    :rtype:
    """
    xdim = rsplice[1] - rsplice[0]
    ydim = csplice[1] - csplice[0]
    part = im[rsplice[0]:rsplice[1], csplice[0]:csplice[1]][:, :, :3]

    # bratio is the ratio of pixels where the RGB value is greater than the inputted min pixel
    # value for black, giving is a measure of how "black" a pixel
    # this then returns whether the ratio is under the threshold, which if it is, means the
    # part is not too black to include in training
    bratio = sum([0 if np.all(part[r][c] > bminpixel) else 1  # check if each pixel is greater than min
                    for r in range(xdim)
                    for c in range(ydim)]) / (xdim * ydim)
    return bratio < bthresh


def labelPart(imname, im, ref, rsplice, csplice, athresh):
    """returns 3-tuple of form (name of partitioned image, numpy array of partitioned image, 0 if nonanomalous; else 1).

    :param:
    :param:
    :param:
    :param:
    :returns:
    :rtype:
    """
    xdim = csplice[1] - csplice[0]
    ydim = rsplice[1] - rsplice[0]
    part = im[rsplice[0]:rsplice[1], csplice[0]:csplice[1]][:, :, :3]
    # aratio is the sum of pixel values in the reference for this splice divided by
    # the total number of pixels, giving us how much of the splice is anomalous
    # ref[r][c] is 1 pixel in the part generated from the inputted row and col splice
    aratio = sum([0 if ref[r][c] == 0 else 1
                    for r in range(rsplice[0], rsplice[1])
                    for c in range(csplice[0], csplice[1])]) / (xdim * ydim)
    return (imname, part, 1) if aratio > athresh else (imname, part, 0)


def saveParts(partedIms, anompath, noanompath):
    """saves partitioned images to disk in up to 2 different locations for anomalous
       versus non-anomalous images.

    :param:
    :param:
    :param:
    :param:
    :returns:
    :rtype:
    """
    for i in range(len(partedIms)):
        for j in range(len(partedIms[i])):
            imname  = partedIms[i][j][0][:-4]
            imarray = partedIms[i][j][1]
            imlabel = partedIms[i][j][2]
            im      = Image.fromarray(imarray)
            impath  = anompath if imlabel==1 else noanompath
            name    = os.path.join(impath, imname + str(i) + "_" + str(j) + ".png")
            im.save(name)


#####################################################
#                                                   #
#                                                   #
# Utils                                             #
#                                                   #
#                                                   #
#####################################################

def underSamp(x0, x1, ratio=[4,1]):
    """return randomized subset of the larger array corresponding to the inputted ratio.

    NOTE  x0 has form [(numpy array, 0)]
    NOTE  x1 has form [(numpy array, 1)]
    NOTE  ratio has form [int1, int2] describing ratio to balance x0 and x1
    NOTE  return has form [[(numpy array, 0)], [(numpy array, 1)]], which is a list
          of lists where each has size corresponding to the inputted ratio.
    """
    whichtosubset = 0 if len(x0) > ((ratio[0] / sum(ratio)) * (len(x0) + len(x1))) else 1
    if whichtosubset==0:
        tosubset = x0
        returnas = x1
    else:
        tosubset = x0
        returnas = x1
    indicescount = len(x1) * ratio[0] if whichtosubset==0 else len(x0)//ratio[0]
    indices      = [i for i in range(len(tosubset))]
    rindices     = random.sample(indices, indicescount)
    return [tup
            for i, tup in enumerate(tosubset)
            if i in rindices], returnas


def getTag(im):
    """return substring of image path string that represents anomaly positions.

    NOTE  different image naming convention will require different tagging
    """
    pattern = re.compile("[pP]\d{2}")
    tag     = [split for split in im.split("_") if pattern.match(split)]
    return tag[0] if tag else ""


def storeIms(ims, directory, tag=None):
    """save numpy images as pngs locally.
    """
    for i in range(len(ims)):
        for j in range(len(ims[i])):
            name = directory + tag + "_" + str(i) + "_" + str(j) + ".png"
            im   = Image.fromarray(ims[i][j])
            im.save(name)


def getIms(path):
    """return pandas dataframe with paths of pngs.
    """
    return pd.DataFrame([im for im in os.listdir(path) if im[-3:]=='png'])


def getImTypes(ims):
    """get image types for use with method subset_imgs.
    """
    return list(set([row[0].split("_")[1] for _, row in ims.iterrows()]))


def toNP(path, im):
    """return image img as numpy array.
    """
    return np.array(Image.open(os.path.join(path, im)))


def openIm(npim):
    """display inline numpy image img.
    """
    plt.figure(figsize = (25,10))
    plt.imshow(npim)


def subsetIms(ims, imtype, leaveout=False, types=
              ['Shirts', 'Paper', 'Laptops', 'Cans', 'Bananas', 'Shoes',
               'Apples', 'Tires', 'AnomalyAbsent', '200', '750']):

    """get subset of images of some type (i.e., string like '200', 'AnomalyAbsent', or 'Apples').

    NOTE  constrains types we can subset by the default value for List[str] in method definition.
    """
    if imtype not in types: raise AttributeError
    return ims[ims[0].str.contains(imtype, case=False) != leaveout]


def getRefIm(imdf):
    """return an image path (name) for a full image that isn't just the anomaly.
    """
    return next(filter(lambda x : "anomaly_only_view" not in x, [tup[1] for tup in list(imdf.itertuples())]))