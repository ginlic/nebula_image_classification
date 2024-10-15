This app 'Nebula Image Classification' allows a user to upload an image of a nebula and predict which of the five categories it belongs to. The primary categories include: emission, reflection, dark, planetary, and supernova.

The dataset for training was sourced using Bing Image Search API and went through multiple iterations of downloading,cleaning (removal of images with invalid formats or duplicated images) and reclassifying. 

The model was built upon EfficientNetB0 using ImageNet weights and max pooling. It achieves an accuracy of 84.2% in training and 83.8% in validation