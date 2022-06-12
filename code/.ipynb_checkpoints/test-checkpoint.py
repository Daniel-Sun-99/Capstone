import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import pickle
import tensorflow as tf
import tensorflow_io as tfio
import os
import sys

sys.path.insert(0, '../VisualizationTools')
import get_data_from_XML, get_gt, getUID, roi2rect, utils, visualization

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping




#Convert a numpy 3d array of a bitmap of an rgb image to grayscale
#using the formula (0.3 * R) + (0.59 * G) + (0.11 * B) 
def rgb_bmp_togray(imgbmp):
    r = imgbmp[:,:,0]
    g = imgbmp[:,:,1]
    b = imgbmp[:,:,2]
    graybmp = np.multiply(0.3*r, 0.59*g)
    graybmp = np.multiply(graybmp, .11*b)
    
    return graybmp

#Intersection over union calculation given two bounding boxes
def IOU(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    xmin2, ymin2, xmax2, ymax2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    x_intersection = 0
    if xmin1 <= xmax2 and xmin2 <= xmax1:
        x_intersection = min(xmax1, xmax2) - max(xmin1, xmin2)
        
    y_intersection = 0
    if ymin1 <= ymax2 and ymin2 <= ymax1:
        y_intersection = min(ymax1, ymax2) - max(ymin1, ymin2)
    
    intersection = x_intersection * y_intersection
    
    if intersection == 0:
        return 0
    
    union = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2) - intersection
    
    return intersection/union
    

    return I / U

dicom_path = '../images/smallTraining/'
annotation_path = '../annot/Annotation/'

anfilenames = os.listdir(annotation_path)
lungfilenames = os.listdir(dicom_path)
num_classes = 4
xtotal = []
ybox = []
yclass = []
for aname in anfilenames:
    #if np.random.random() > .1:
    #    continue
    anpath = annotation_path + aname
    lungpath = dicom_path + "Lung_Dx-" + aname
    if not os.path.isdir(lungpath):
        print("missing: ", lungpath)
        continue
    lungs = getUID.getUID_path(lungpath)
    annotations = get_data_from_XML.XML_preprocessor(anpath, num_classes=num_classes).data
    for k, v in annotations.items():
        
        key = k[:-4]
        if key not in lungs:
            print("missing key: ", k)
            continue
        dcm_path, dcm_name = lungs[k[:-4]]
        matrix, frame_num, width, height, ch = utils.loadFile(dcm_path)
        img_bitmap = utils.MatrixToImage(matrix[0], ch)
        xbmp = img_bitmap
        if len(img_bitmap.shape) > 2: #assume bitmap is rgb
            xbmp = rgb_bmp_togray(img_bitmap)
       
    
        #xbmp = np.resize(xbmp, (240, 240))
        xbmp = np.resize(xbmp, (224, 224))
        
        
        xtotal.append(np.expand_dims(xbmp, -1).repeat(3, -1))

        #extract xmin, ymin, xmax, ymax in that order
        ybox.append(np.array([v[0][0], v[0][1], v[0][2], v[0][3]]))
        yclass.append(np.array([v[0][4], v[0][5], v[0][6], v[0][7]]))
    

        
        
        
xtotal = np.array(xtotal)
ybox = np.array(ybox)
yclass = np.array(yclass)

X_train, X_test, y_train_box, y_test_box, y_train_class, y_test_class = train_test_split(xtotal, ybox, yclass, random_state = 2022)

es = EarlyStopping(monitor = 'loss', patience = 10)


effNet = EfficientNetB0(include_top=False, weights='imagenet')

effNet.trainable = False

enModel = Sequential()
enModel.add(effNet)
enModel.add(GlobalAveragePooling2D())
enModel.add(Dropout(0.2))
enModel.add(Dense(4))

enModel.compile(optimizer = 'adam', loss = 'mse')
enModel.fit(X_train, y_train_box, epochs = 1, batch_size = 1, verbose = 2, callbacks = es)