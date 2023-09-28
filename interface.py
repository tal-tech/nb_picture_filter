# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import tensorflow as tf
import cv2
import time


CLASSIFY_MODEl_PATH = "./models/anomaly_det_model_v1.0.0.pb"
INPUT_PLACEHOLDER = "Placeholder:0"
OUTPUT_PLACEHOLDER = "logits:0"

class anomaly_check(object):

    def __init__(self):
        
        self.class_names = ['bad', 'normal', 'black']
        # load model.
        config = self._get_sess_config()
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=config)
        with self.g.as_default():
            output_graph_def = tf.GraphDef()
            with open(CLASSIFY_MODEl_PATH, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess.run(tf.global_variables_initializer())
            self.img_input = self.sess.graph.get_tensor_by_name(INPUT_PLACEHOLDER)
            self.pred_result = self.sess.graph.get_tensor_by_name(OUTPUT_PLACEHOLDER)

        print('load anomaly classify model')

    def _get_sess_config(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.02)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        return config

    def preprecess(self, image):

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image
    
    @staticmethod
    def check_black(image):
        b = cv2.split(image)[0]
        g = cv2.split(image)[1]
        r = cv2.split(image)[2]
        b_arr_var = np.var(b)
        g_arr_var = np.var(g)
        r_arr_var = np.var(r)
        total_var = b_arr_var + g_arr_var + r_arr_var

        if total_var < 10:
            return True
        else:
            return False

    def run(self, image):
        """
            main entrance.
        """
        tmp_image = image.copy()

        if self.check_black(tmp_image):
            pred = self.class_names.index('black')
        else:
            tmp_image = self.preprecess(tmp_image)
            result = self.sess.run(fetches=self.pred_result, feed_dict={self.img_input: tmp_image})
            pred = np.argmax(result)

        return self.class_names[pred]


temp_detector = anomaly_check()


def detector(img):
    result = temp_detector.run(img)
    return result
