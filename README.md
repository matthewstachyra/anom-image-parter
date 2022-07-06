anom-image-parter is a Python library to prepare image for anomaly detection tasks.
-----------------

Originally developed for an anomaly detection problem I was solving as part of an internship. The images were xray images of trucks with cargos.

Example of algorithm and documentation:

```
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

```
