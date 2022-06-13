# Capstone Project of General Assembly Data Science Immersive Bootcamp -- Daniel Sun

## Overview
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For the culmination of my time at General Assembly's Data Science Immersive program, I decided on training and fitting a model to predict the location of cancerous masses within a given image. Within this project I utilized my knowledge of convolutional neural networks, data visualisation, and model validation and analysis. The dataset chosen for this project is a public dataset from the Cancer Imaging Archive under the name [A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216#70224216bcab02c187174a288dbcbf95d26179e8). The packages used consist of numpy, matplotlib, tensorflow's keras, and sklearn.

------

## The Dataset
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The entire dataset is a culmination of a number of CT and PET scans of lung cancer subject; approximately 130gb. The image files are uploaded in Dicom, .dcm, format as this is a common file type for CT and PET scans. Along with the images, a second folder of annotations containing cancer classification and coordinates for a bounding box indicating the location of a cancerous mass within specific image files, and a useful folder of pyscript files to pull image data and annotation data together, was uploaded to the Cancer Imaging Archive. Due to hardware restriction, a subset of this data was taken while ensuring that each cancer class, of which there were 4, were taken from the database.

------

## General Procedure
### Data Aquisition
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As mentioned in the overview, the dataset was aquired from the Cancer Imaging Archive with the help of a tool named the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+the+NBIA+Data+Retriever+7.7). The archive alows for a full download or a query to select a subset of files. This downloads the images in their .dcm format and annotations in .xml. To pull the image and annotation data from their respective files, pycripts included in the archive by the project uploaders were ultilized.

### Model Creation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After the image data was pulled down and their corresponding annotated bounding box coordinates were paired, the data was split into training and testing sets. The data was then passed through a feed forward convolutional neural net to predict bounding box coordinates, e.g, the position of a cancerous mass within a given image. Two other feed forward covolutional neural net were trained to predict the class of cancer in the image using the image data in its entirety and just the bounding box coordinates. Various parameters and image aumentation, within my hardware's constraints, were tested to try and improve the model's performance.

### Model Validation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model validation was performed on the test set during the train test split, as well as an entirely new subset of data taken from the same archive. To measure the accuracy of the bounding box prediciton, the IOU (intersection over union) metric was calculated for each prediction. The F1 score was calculated to assess the accuracy of the two classification models.

### Data Visualisation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;All images were created and visualised using cv2 and matplotlib and saved as png files.

------

## Conclusion
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Within the training and test set, the models trained, one for predicting bounding box locations and two for predicting cancer class, appeared to work fairly well. With approximately 75% of the predicted bounding boxes sharing some overlap with the actual bounding boxes, approximately 37% on average, and both classification models having over 70% accuracy, it appears that the simple convolutional neural nets created could semi-accurately predict the class and location of lung cancer tumors in a given ct scan of a chest. However, the models, in reality, are overfit. Taking a fresh subset of images from the same database, the accuracy drops to around 40% on both models with significant trouble identifying cancer types B and E. The bounding box predictions hold a similar IOU of 30%, however, the percentage of predictions that do share an overlap with the true bounding box drops to around 30%. As such, the models are not ready for production. Further tuning of the models was inhibited by my systems hardware and memory storage. Future steps include using a prebuilt neural net like EfficientNet or ResNet50V2 to use as a base and building on top of it. 