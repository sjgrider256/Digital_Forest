# Abstract

Digital_Forest is a deep learning pipeline for species detection and classification of mixed forest canopies using UAV imagery. This workflow combines transfer learning of two pretrained convolutional neural networks (CNNs) to predict tree species and forest composition from high-resolution RGB imagery. The procedures below outline training and evaluation of a model to detect Eastern Hemlock (Tsuga canadensis), currently threatened by invasive Hemlock Woolly Adelgid (Adelges tsugae) in the eastern United States. This deep learning system can aid conservation efforts by providing geo-referenced data and health metrics on individual stands across large forest plots with a single flyover. Furthermore, the workflow outlines a reliable system for creating forest inventories with extreme accuracy, and can be extended to other species using similar training pipelines. This repository is intended as an archive for predictive forest composition models, where collaborators are invited to contribute datasets and models to expand its utility for various species across diverse ecosystems.


![Screenshot](images/Screenshot%202025-06-05%20095418.png)

# Dependencies

**[DeepForest](https://github.com/weecology/DeepForest)** ([Weinstein et al., 2020](https://doi.org/10.1038/s41597-020-0449-9)):  DeepForest is a deep learning model built on the RetinaNet architecture, designed to delineate individual tree canopies from RGB imagery. In this workflow, DeepForest is used to extract images of individual trees from the orthomosaic. The extracted images retain their geospatial data and are used as input for our classification model classification.

**Packages**:
- torch
- numpy
- pandas
- matplotlib
- rasterio
- shapely
- geopandas

---

**[EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)**([Tan & Le, 2019](https://arxiv.org/abs/1905.11946)):  
  EfficientNet is a family of CNN developed by Mingxing Tan and Quoc V. Le at Google AI. The models scale depth, width, and resolution with fewer parameters and FLOPs. The base model (EfficientNetB0) is used in this project for binary classification (e.g., Hemlock vs. Other) of tree species from image crops with transfer learning.

**Packages**:
- tensorflow
- keras
- opencv-python
- scikit-learn



# 📁 Datasets

Access datasets here --> [**Digital_Forest(Eastern_Hemlock)**](https://drive.google.com/drive/folders/1v7P8ayvgNeTtqQJLFxYiCn26fgUE1_lM)

&nbsp; &nbsp; &nbsp; Copy the shared folder to your Google Drive. 

## Training
-This dataset contains images collected from multiple flights to reduce overfitting. Training data categories are balanced in size so that the CNN does not favor feature learning for one category over the other. Assigning class weights can improve an imbalanced dataset, but if the imbalance is significant enough, learning will be skewed regardless of applied weights. 
   
## Prediction 
-This is the full dataset extracted from the orthomosaic. 

# Process

- Copy the shared Google Drive folder into your own drive.

### Training a model from scratch with transfer learning
Explain steps and resources in Google colab notebook (step 1 to prediction). Dont forget combined metadata file, the
saved weights, etc.

2. Open the Google Colab notebook from this repository.
3. Mount Google Drive and import necessary libraries.
4. Load the training data and apply preprocessing (e.g., resizing, normalization).
5. Initialize EfficientNetB0 with include_top=False and add a custom classification head.
6. **Train** using the training dataset and validate on the held-out validation set.
7. **Save** the model weights and accuracy logs to your Google Drive.
8. **Export** predictions as CSV files that include image name, predicted label, and probability.

Combined metadata file (merged during preprocessing) links each image to its geographic location for GIS visualization.

### Prediction
if you want to use our pretrained model to detect eastern hemlock on your own dataset, skip to step ______ and run predictions on the full dataset from the saved model (in google drive folder). 

To use our pretrained model for Eastern Hemlock detection on your own dataset:

- Skip to **Step 6** in the Colab notebook and load the saved model weights.
- Run predictions on the full prediction dataset extracted by DeepForest.
- Results will be saved with predicted labels and probabilities.
### Output
Model predictions are merged with the combined metadata file, which contains image filenames, lat/lon coordinates, and bounding box information. The final output is a CSV that can be imported into QGIS or other GIS software to visualize detected species across the forest plot.

### Building your own dataset
Drones. Flyover. 2cm/pixel GGSD. creating the orthmosiac (include picture) Image extraction (deep_foest)

- **Flight**: Conduct UAV flyovers at ~2 cm/pixel GSD using RGB cameras. Multiple passes improve coverage.
- **Orthomosaic**: Use software like WebODM or Pix4D to stitch raw imagery into a georeferenced orthomosaic.
- **Image Extraction**: Run DeepForest on the orthomosaic to detect and extract individual tree crowns. These sub-images form your training and prediction dataset.

---

Contributions are welcome. Submit pull requests or issues for dataset formats, model integration, or results interpretation.
