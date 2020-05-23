#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:48:00 2020

@author: Seetharam/setharram@gmail.com
"""

import cv2 as cv
import tensorflow as tf
import numpy as np
import os


datst_pt = '/home/ram/Documents/opencv/face-recognition-opencv/dataset'
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
dataset_labels = ['alan_grant','claire_dearing','ellie_sattler','ian_malcolm','john_hammond','owengrady']
tf_model = '/home/ram/Documents/opencv/face-recognition-opencv/zurassic_clasfy-tes/converted_model.tflite'

def classify_image(image,modl,labels):
     # Load TFLite model and allocate tensors.
     interpreter = tf.lite.Interpreter(model_path=modl)
     interpreter.allocate_tensors()

     # Get input and output tensors.
     input_details = interpreter.get_input_details()
     output_details = interpreter.get_output_details()

     outpt = cv.resize(image,(224,224))/255.0
     outpt = np.array(outpt,dtype=np.float32)
     interpreter.set_tensor(input_details[0]['index'], [outpt])
     interpreter.invoke()

     # Use `tensor()` in order to get a pointer to the tensor.
     tflite_results = interpreter.get_tensor(output_details[0]['index'])
     print(tf.keras.layers.Softmax(tflite_results))
     predicted_ids = np.argmax(tflite_results, axis=-1)
     print(labels[predicted_ids[0]])
     return labels[predicted_ids[0]]
     
for fil in os.listdir(datst_pt):
     for imag in os.listdir(datst_pt+'/'+fil):
          
          # load the input image and convert it from BGR to RGB
          image = cv.imread(datst_pt+'/'+fil+'/'+imag)
          gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
          
          faces = face_cascade.detectMultiScale(image, 1.1, 4)
          
          nam = classify_image(image, tf_model, dataset_labels)
          
          for (x,y,w,h) in faces:
               cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
               cv.putText(image, nam,\
                      (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
          
          image = cv.resize(image,(512,512))
          cv.imshow("Image", image)
          
          if cv.waitKey(0) == ord('q'):
               break

cv.destroyAllWindows()


