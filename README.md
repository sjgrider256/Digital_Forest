# Abstract

Digital_Forest is a deep learning pipeline for species detection and classification of mixed forest canopies using UAV imagery. This workflow combines transfer learning of two pretrained convolutional neural networks (CNNs) to predict tree species and forest composition from high-resolution RGB imagery. The procedures below outline training and evaluation of a model to detect Eastern Hemlock (Tsuga canadensis), currently threatened by invasive Hemlock Woolly Adelgid (Adelges tsugae) in the eastern United States. This deep learning system can aid conservation efforts by providing geo-referenced data and health metrics on individual stands across large forest plots with a single flyover. Furthermore, the workflow outlines a reliable system for creating forest inventories with extreme accuracy, and can be extended to other species using similar training pipelines. This repository is intended as an archive for predictive forest composition models, where collaborators are invited to contribute datasets and models to expand its utility for various species across diverse ecosystems.


![Screenshot](images/Screenshot%202025-06-05%20095418.png)

# Dependencies
The workflow outlined below is comprised of two existing computer vision models: DeepForest & EfficientNet.

Attribute authors and summarize/describe them briefly. Describe their purpose (what they do) and their functionality in my model: deepforest is used to delineate forest canopies with bounding boxes, which can then be used to extract images with geography data from the original orthomosaic and Efficiant Net is used to classify/categorize/predict tree species.  Then briefly describe how they operate and their dependencies (python libraries)


## 📁 Datasets

Access datasets here --> [**Digital_Forest(Eastern_Hemlock)**](https://drive.google.com/drive/folders/1v7P8ayvgNeTtqQJLFxYiCn26fgUE1_lM)

&nbsp; &nbsp; &nbsp; Copy the shared folder into your Drive as a root directory. 

### Training
This dataset contains images collected from multiple flights to reduce overfitting.

Training data categories are balanced in size so that the CNN does not favor feature learning for one category over the other. Assigning class weights can improve an imbalanced dataset, but if the imbalance is significant enough, learning will be skewed regardless of applied weights. 
   
### Prediction 
This is the full dataset extracted from the orthomosaic. 

## Proceess

copy the shared folder to your google drive. Access the Google Colab notebook


### Training a model from scratch with transfer learning
Explain steps and resources in Google colab notebook (step 1 to prediction). Dont forget combined metadata file, the
saved weights, etc.


### Prediction
if you want to use our pretrained model to detect eastern hemlock on your own dataset, skip to step ______ and run predictions on the full dataset from the saved model (in google drive folder). 
### Output
merging predicts with combined meta_data (where that comes from) and visualizing in a GIS. 
### Building your own dataset
Drones. Flyover. 2cm/pixel GGSD. creating the orthmosiac (include picture) Image extraction (deep_foest)
