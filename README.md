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

-Load the training data. Resize images to match model input resolution (e.g., EfficientNetB0 requires 224x224 pixel input). Batch size determines the number of images being processed for each step. Split the dataset into training & validation subsets (e.g., 80/20). 
-Apply augmentation to images for regularization and optimize training with prefetching
-Initialize EfficientNetB0. Set "include_top=False" for transfer learning (freeze learning on the base layers-previously learned from ImageNet training). Use class weights for imbalanced datasets
-Train the model. Set epochs = 20 to start. The model will backpropagate (adjust weights) based on the loss function given at the end of each epoch. 
-Save the model weights and accuracy logs to your Google Drive.
-Evaluate the model. A well-trained model will show little difference in accuracy between training and validation datasets, with a gradual increase. Additionally, validation loss should decrease at a comparable rate to training loss.

![EfficientNetB0overfitting](images/Screenshot%202025-05-02%20165740.png)

An overfit model (meaning the model memorized the data instead of learning features) will show little to no improvement on validation

![EfficientNetB0overfitting](images/Screenshot%202025-05-02%20165740.png)


### Prediction
To make predictions from our pretrained model (located in Google Drive ) for Eastern Hemlock detection on your own dataset:

- Skip to Step 6 in the Colab notebook and load the saved model weights.
- Run predictions on the full prediction dataset extracted by DeepForest.
- Results will be saved with predicted labels and probabilities.
  
### Output
Model predictions are merged with the combined metadata file, which contains image filenames, lat/lon coordinates, and bounding box information obtained from the original DeepForest predictions. The final output is a CSV that can be imported into QGIS or other GIS software to visualize detected species across the forest plot.

### Building your own dataset

**Flight**: Viable datasets must contain high resolution RGB images (~2 cm/pixel GSD). There are many commercially available drones capable of conducting grid surveys. Third-party software such as MapPilot Pro and Litchi perform reliably with most DJI models. Premium DJI models have native mapping software. For optimal results, conduct surveys at constant altitude above ground level (AGL), and ensure overlap along the path exceeds 85%. Across-path overlap should be greater than 80%.

For large forest plots ( > 20 acres), DSLR cameras with 24+ megapixel sensors attached to custom drones can reduce the number of required images, extend flight times, and increase mission efficiency. For this project, we constructed a custom Pixhawk Hexacopter, fitted with a 24 megapixel Sonny DSLR camera. Flight plans were made in MissionPlanner at 95m AGL with 90% overlap along path and 85% across, yielding a calculated ground sampling distance (GSD) of ~2cm/pixel. Images were georeferenced using CAM messaging via ArduPilot. 

![Remote Sensing Drone performing survey for eastern hemlock](images/DJI_0231.jpg)

**Orthomosaic Generation**: Stitch raw imagery into a georeferenced orthomosaic before delineating the canopy with DeepForest. This ensures that extracted images contain geographical data. WebODM is a free, open-source Docker program capable of rendering high-resolution orthomosaics. For large datasets (1000+ images), consider deploying an AWS Elastic Container to process your orthomosaic with GPU or TPU capability. Paid services like Pix4D and Agrisoft are easy to contain many additional features.

![RGBOrthomosaic](images/Screenshot%2025-05-15%104932.png)

**Tree Crown Delineation**: With the orthomosaic ready, use the DeepForest Python library to delineate individual tree crowns from the forest canopy. This will produce bounding boxes/geometries around each tree crown. Adjust patch size and overlap parameters for best fit. For optimal results, annotate your predictions with additional training.

**Image Extraction**:  Open your predictions in QGIS or another GIS program, select your target class specimens by labeling them (1 = target class, 0 = other), and then export each class as a separate shapefile. Then, overlay the classified shapefiles onto your orthomosaic and crop images from the bounding boxes for each tree produced by your DeepForest predictions. Shapefile geometry must be converted from geographical coordinates to pixel coordinates. Save pixel coordinates (xmin, ymin, xmax, & ymax) as metadata attached to each file in order to match species predictions from the EfficientNet classification to your original orthomosaic. Choose your framework of preference (Python, C+++, Java, etc.) for this step. 

---

Contributions are welcome. Submit pull requests or issues for dataset formats, model integration, or results interpretation.
