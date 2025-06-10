# Abstract

Digital_Forest is a deep learning pipeline for species detection and classification of mixed forest canopies using UAV imagery. This workflow combines transfer learning of two pretrained convolutional neural networks (CNNs) to predict tree species and forest composition from high-resolution RGB imagery. The procedures below outline training and evaluation of a model to detect Eastern Hemlock (Tsuga canadensis), currently threatened by invasive Hemlock Woolly Adelgid (Adelges tsugae) in the eastern United States. This deep learning system can aid conservation efforts by providing geo-referenced data and health metrics on individual stands across large forest plots with a single flyover. Furthermore, the workflow outlines a reliable system for creating forest inventories with extreme accuracy, and can be extended to other species using similar training pipelines. This repository is intended as an archive for predictive forest composition models, where collaborators are invited to contribute datasets and models to expand its utility for various species across diverse ecosystems.


![Screenshot](images/Screenshot%202025-06-05%20095418.png)

# Dependencies

**[EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)**([Tan & Le, 2019](https://arxiv.org/abs/1905.11946)):  
  EfficientNet is a family of CNN developed by Mingxing Tan and Quoc V. Le at Google AI. The models scale depth, width, and resolution with fewer parameters and FLOPs. The base model (EfficientNetB0) is used in this project for binary classification (e.g., Hemlock vs. Other) of tree species from image crops with transfer learning.

**Packages**:
- tensorflow
- keras
- opencv-python
- scikit-learn

---

**[DeepForest](https://github.com/weecology/DeepForest)** ([Weinstein et al., 2020](https://doi.org/10.1038/s41597-020-0449-9)):  DeepForest is a deep learning model built on the RetinaNet architecture, designed to delineate individual tree canopies from RGB imagery. In this workflow, DeepForest is used to extract images of individual trees from the orthomosaic. The extracted images retain their geospatial data and are used as input for our classification model classification.

**Packages**:
- torch
- numpy
- pandas
- matplotlib
- rasterio
- shapely
- geopandas

# 📁 Datasets

Access datasets here --> [**Digital_Forest(Eastern_Hemlock)**](https://drive.google.com/drive/folders/1v7P8ayvgNeTtqQJLFxYiCn26fgUE1_lM)

&nbsp; &nbsp; &nbsp; Copy the shared folder to your Google Drive. 

**Training:** This dataset contains images collected from multiple flights to reduce overfitting. Training data categories are balanced in size so that the CNN does not favor feature learning for one category over the other. Assigning class weights can improve an imbalanced dataset, but if the imbalance is significant enough, learning will be skewed regardless of applied weights. 
   
**Prediction:** This is the full dataset extracted from the orthomosaic. 

# Process

- Copy the shared Google Drive folder into your own drive.
- Open the Google Colab notebook from this repository.
- Mount Google Drive and import necessary libraries.

### Training a model from scratch with transfer learning
Explain steps and resources in Google colab notebook (step 1 to prediction). Dont forget combined metadata file, the
saved weights, etc.


4. Load the training data and apply preprocessing (e.g., resizing, normalization).
5. Initialize EfficientNetB0 with include_top=False and add a custom classification head.
6. **Train** using the training dataset and validate on the held-out validation set.
7. **Save** the model weights and accuracy logs to your Google Drive.
8. **Export** predictions as CSV files that include image name, predicted label, and probability.

Combined metadata file (merged during preprocessing) links each image to its geographic location for GIS visualization.

### Prediction
To make predictions from our pretrained model (located in Google Drive ) for Eastern Hemlock detection on your own dataset:

- Skip to Step 6 in the Colab notebook and load the saved model weights.
- Run predictions on the full prediction dataset extracted by DeepForest.
- Results will be saved with predicted labels and probabilities.
  
### Output
Model predictions are merged with the combined metadata file, which contains image filenames, lat/lon coordinates, and bounding box information. The final output is a CSV that can be imported into QGIS or other GIS software to visualize detected species across the forest plot.

### Building your own dataset

**Flight**: Viable datasets must contain high resolution RGB images (~2 cm/pixel GSD). There are many commercially available drones capable of conducting grid surveys. Third-party software such as MapPilot Pro and Litchi perform reliably with most DJI models. Premium DJI models have native mapping software. For optimal results, conduct surveys at constant altitude above ground level (AGL), and ensure overlap along the path exceeds 85%. Across-path overlap should be greater than 80%.

For large forest plots ( > 20 acres), DSLR cameras with 24+ megapixel sensors attached to custom drones can reduce the number of required images, extend flight times, and increase mission efficiency. For this project, we constructed a custom Pixhawk Hexacopter, fitted with a 24 megapixel Sonny DSLR camera. Flight plans were made in MissionPlanner at 95m AGL with 90% overlap along path and 85% across, yielding a calculated ground sampling distance (GSD) of ~2cm/pixel. Images were georeferenced using CAM messaging via ArduPilot. 

![Remote Sensing Drone performing survey for eastern hemlock](images/DJI_0231.jpg)

**Orthomosaic Generation**: Stitch raw imagery into a georeferenced orthomosaic before delineating the canopy with DeepForest. This ensures that extracted images contain geographical data. WebODM is a free, open-source Docker program capable of rendering high-resolution orthomosaics. For large datasets (1000+ images), consider deploying an AWS Elastic Container to process your orthomosaic with GPU or TPU capability. Paid services like Pix4D and Agrisoft are easy to contain many additional features.

**Tree Crown Delineation**: With the orthomosaic ready, use the DeepForest Python library to delineate individual tree crowns from the forest canopy. This will produce bounding boxes/geometries around each tree crown. Adjust patch size and overlap parameters for best fit. For optimal results, annotate your predictions with additional training.

**Image Extraction**:  Open your predictions in QGIS or another GIS program, select your target class specimens by labeling them (1 = target class, 0 = other), and then export each class as a separate shapefile. Then, overlay the classified shapefiles onto your orthomosaic and crop images from the bounding boxes for each tree produced by your DeepForest predictions. Shapefile geometry must be converted from geographical to pixel coordinates. Save pixel coordinates (xmin, ymin, xmax, ymax) as metadata attached to each filename to match species predictions from EfficientNet classifications to the orthomosaic. Choose your framework of preference (Python, C+++, Java, etc.) for this step. 

---

Contributions are welcome. Submit pull requests or issues for dataset formats, model integration, or results interpretation.
