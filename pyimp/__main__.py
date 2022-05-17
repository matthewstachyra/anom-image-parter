"""run pyImp from here
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import pyimp

if __name__ == "__main__":

    # NOTE  update below as variables to control behavior
    SUBTYPE     = 'Apples'      # which subtype of images to grab, if any
    MODE        = 'feature'     # either 'default' or 'feature'

    DIR         = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    PATH        = os.path.join(DIR, 'images', 'partitioned', MODE)
    ANOMPATH    = os.path.join(PATH, 'anom')
    NOANOMPATH  = os.path.join(PATH, 'noanom')
    ANOMFILE    = os.path.join(ANOMPATH, str(SUBTYPE + str(1) + ".npy"))
    NOANOMFILE  = os.path.join(NOANOMPATH, str(SUBTYPE + str(0) + ".npy"))

    #####################################################
    #                                                   #
    #                                                   #
    # Prepare images                                    #
    #                                                   #
    #                                                   #
    #####################################################

    if not os.path.exists(NOANOMFILE) and not os.path.exists(ANOMFILE):
        imdf      = pyimp.getIms(PATH)
        print(imdf.head())
        imdf      = pyimp.subsetIms(imdf, SUBTYPE)
        reference = pyimp.buildReference(PATH, imdf)
        refim     = pyimp.getRefIm(imdf)
        splices   = pyimp.createSplices(PATH, refim, mode='feature', dim=64, k=4)
        partimgs  = pyimp.imPartition(PATH, imdf, reference, splices)

        # NOTE: refactor below to be cleaner.
        part0, part1 = [], []
        parts = []
        for p in partimgs:
            if isinstance(p, list):
                parts.append(p)
                for i in range(len(p)):
                    if p[i][2]==0:
                        part0.append((p[i][1], p[i][2]))
                    elif p[i][2]==1:
                        part1.append((p[i][1], p[i][2]))
        np.save(NOANOMFILE, part0, allow_pickle=True)
        np.save(ANOMFILE, part1, allow_pickle=True)
        pyimp.saveParts(parts, ANOMPATH, NOANOMPATH)


    #####################################################
    #                                                   #
    #                                                   #
    # Build dataset                                     #
    #                                                   #
    #                                                   #
    #####################################################

    part0 = np.load(NOANOMFILE, allow_pickle=True)
    part1 = np.load(ANOMFILE, allow_pickle=True)

    part0, part1 = pyimp.underSamp(part0, part1)                                    # 80:20 distribution, by default
    xfunc = lambda x : (np.asarray(x[0], dtype="float") /                           # normalize and flatten
                        np.linalg.norm(np.asarray(x[0], dtype="float"))).flatten()
    yfunc = lambda x : np.asarray(x[1])
    X = np.concatenate((
                np.asarray(list(map(xfunc, part0))),
                np.asarray(list(map(xfunc, part1)))), axis=0)
    y = np.concatenate((
                np.asarray(list(map(yfunc, part0))),
                np.asarray(list(map(yfunc, part1)))), axis=0)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

    #####################################################
    #                                                   #
    #                                                   #
    # Support Vector Machine                            #
    #                                                   #
    #                                                   #
    #####################################################

    clf  = svm.SVC(kernel='linear')
    ypred = clf.fit(Xtrain, ytrain).predict(Xtest)
    print(classification_report(ytest, ypred))
    print(confusion_matrix(ytest, ypred))
    print(accuracy_score(ytest, ypred))


    grid = {'C':[0.1,1,100,1000],
            'kernel':['rbf','poly', 'linear'],
            'degree':[4,5,6],
            'gamma': [1, 0.1, 0.01]}
    grid = GridSearchCV(svm.SVC(), grid, refit = True)
    grid.fit(Xtrain, ytrain)
    print(classification_report(ytest, grid.best_estimator_.predict(Xtest)))
    print(confusion_matrix(ytest, grid.best_estimator_.predict(Xtest)))
    print(grid.score(Xtest, ytest))
    print(grid.best_params_)
