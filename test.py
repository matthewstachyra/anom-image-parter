import os
import pandas as pd
import pytest
from pyimp import pyimp


@pytest.fixture
def path(path="images"):
    '''return full path to images given a relative path to the images.
    '''
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


@pytest.fixture
def df(path):
    '''return a dataframe with any images stored at the path using getIms().
    '''
    return pyimp.getIms(path)


@pytest.fixture
def ref(path, df):
    '''return reference dictionary for use in imPartition().
    '''
    return pyimp.buildReference(path, df)


def test_getIms(path, df):
    '''test getIms() by asserting some dataframe is returned.
    '''
    print(df.head())
    assert not df.empty


def test_getRefIm(df):
    '''test getRefIm() by asserting an image is returhed and that it contains only the anomaly.
    '''
    ref = pyimp.getRefIm(df)
    assert ref and "anomaly_only_view" not in ref


#def test_buildreference(path, df, ref):
#    '''test buildReference() by asserting whether a dictionary with a reference image for every tag is returned.
#    '''
#    # verify ref is a dictionary
#    #TODO
#
#    # count number of tags
#    #TODO
#
#    # check whether ref has as many keys as there are tags
#    #TODO
#
#
#def test_subsetIms(df):
#    '''test subsetIms() by asserting the items in the returned dataframe all contain the inputted substring in the path.
#    '''
#    #TODO
#
#
#    # default leaveout=False
#    for t ['Shirts', 'Paper', 'Laptops', 'Cans', 'Bananas', 'Shoes', 'Apples', 'Tires', 'AnomalyAbsent', '200', '750']:
#        subset = pyimp.subsetIms(df, t)
#        assert 
#
#
#def test_imPartition_square():
#    #TODO
#
#
#def test_imPartition_variable():
#    #TODO
