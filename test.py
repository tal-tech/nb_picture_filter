# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import tensorflow as tf
import cv2
import time
import glob
from interface import anomaly_check

# 将违规图片放到data目录下
test_data_path = "./data/*.jpg"
tmp_img_list = glob.glob(test_data_path)

tmp_detector = anomaly_check()

for tmp_img in tmp_img_list:
    st_time = time.time()
    tmp_img_path = tmp_img
    tmp_img_src = cv2.imread(tmp_img_path)

    tmp_result = tmp_detector.run(tmp_img_src)
    print("{} - {}".format(tmp_result,time.time()-st_time))
