import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/Users/chervjay/.kaggle/competitions/dogs-vs-cats-redux-kernels-edition/train.zip'
TEST_DIR = '/Users/chervjay/.kaggle/competitions/dogs-vs-cats-redux-kernels-edition/test.zip'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = f"dogsvscats-{LR}-{'2conv-basic'}.model"