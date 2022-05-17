import os
import pandas as pd
import pytest
from pyimp import pyimp


@pytest.fixture
def path():  # path to images
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")


@pytest.fixture
def df(path):  # dataframe of image names
    return pyimp.getIms(path)


def test_getIms(path, df):
    print(df.head())
    assert not df.empty
