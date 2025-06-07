# Digital_Forest
Overview. Digitized Forest Inventories. What is Digital Forest? What is it capable of

![Digital_Forest_Output](https://myoctocat.com/assets/images/base-octocat.svg)
how it works: computer vision/cv model 

Purpose. Invitation for collaborators: An archive for deep learning forest inventory models. Implicaitons for conservation

Dependencies
Deepforest
other libraries


## 📁 Datasets

Access datasets here --> [**Digital_Forest(Eastern_Hemlock)**](https://drive.google.com/drive/folders/1v7P8ayvgNeTtqQJLFxYiCn26fgUE1_lM)

&nbsp; &nbsp; &nbsp; Copy the shared folder into your Drive as a root directory. 

### Training
This dataset contains images collected from multiple flights to reduce overfitting.

Training data categories are balanced in size so that the CNN does not favor feature learning for one category over the other. Assigning class weights can improve an imbalanced dataset, but if the imbalance is significant enough, learning will be skewed regardless of applied weights. 
   
### Prediction 
This is the full dataset extracted from the orthomosaic. 

### Combined metadata file
Where did it come from and why is it important? 
### Model
saved weights

### GeoTIFF
Orthomosaic: how it was made.

img of tiff file


## Proceess

copy the shared folder to your google drive. Access the Google Colab notebook

### Training a model from scratch with transfer learning
Credit the keras/google authors. Explain the concepts of this CNN for someone trying to train a model for a new species. 
### Prediction
if you want to use our pretrained model to detect eastern hemlock on your own dataset, skip to step ______ and run predictions on the full dataset from the saved model (in google drive folder). 
### Output
merging predicts with combined meta_data (where that comes from) and visualizing in a GIS. 
### Building your own dataset
Drones. Flyover. 2cm/pixel GGSD. Image extraction (deep_foest)
