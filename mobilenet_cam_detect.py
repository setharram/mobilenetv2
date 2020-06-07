#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:50:45 2020

@author: Seetharam/setharram@gmail.com
"""

import cv2 as cv
import tensorflow as tf
import numpy as np

cap = cv.VideoCapture(0)
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height


if not cap.isOpened():
    print("Cannot open camera")
    exit()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# init tensorflow 
dataset_labels = ['alan_grant','claire_dearing','ellie_sattler','ian_malcolm','john_hammond','owengrady']
tf_model = 'mobilenetV2.tflite'

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
     # calculating probabilities
     tf_exp = np.exp(tflite_results - np.max(tflite_results))
     prob = tf_exp/tf_exp.sum()
     # extract text labels
     predicted_ids = np.argmax(tflite_results, axis=-1)

     return labels[predicted_ids[0]],prob[0][predicted_ids[0]]


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    image = cv.flip(frame, 1)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # call tensorflow lite interpreter
    name,prob = classify_image(image, tf_model, dataset_labels)
    prob = str(round(prob,2))
    print(name,prob)
    
    for (x,y,w,h) in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv.putText(image, name,\
                       (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv.putText(image, prob,\
                       (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # Display the resulting frame
    cv.imshow('face detection', image)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()   
cv.destroyAllWindows()