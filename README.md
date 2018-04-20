# cellSeg
Kaggle Data Science Bowl 2018 (https://www.kaggle.com/c/data-science-bowl-2018)

Our submission to the Kaggle Data Science bowl. Reached a top 5% entry in the final leader board. The approach consists of two Unets. One is responsible for predicting cell bodies the other one is used to get boundaries between cells. These are substracted from the cell bodies when finding individual cells using the watershed approach. **Data augmentation** is flips, rotations and zooming into random 256x256 areas of the mostly larger cell images. We used one external dataset for the colored images. For **pre-processing** we used histogram equivalization and transformation from RGB into CIELAB color space for colored images.

Very important was the rescaling test images in order to have similar average size for the predicted cells. This is done in the script `eval_model_tiling.py`. We tile the test image for small cells and fuse the predicted tiles back together for the final prediction. Another important performance boost was the seperation of colored and grayscale into seperate models. 

Possible extensions that we did not manage in time:
 * Test further data augmentation: Intensity, Gaussian noise
 * Use Mask-RCNN (possibly combine with unet)

## Installation

Put all data into the folder `../inputs`. Do the splitting for the training and test set into colored and grayscale images:

```python
loadData.split_data()
```

To process the test set labels and the 2nd stage test set run

```python
loadData.split_data_new_data()
```

Run `write_masks_test.py` to convert the test labels in the csv file into the training set structure.

## Training

Run `train_model.py` to train with the two unets. Set `useExternal = 0` since we currently do not provide the script to process the colored external data set.

## Evaluation

For the evaluation, run `eval_models_tiling.py`.
