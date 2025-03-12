# Part I:
    - Goal: Train a classifier to predict wsi class

## Solution:

### High level:
    open slides and divide into tiles in the highest resolution, remove tiles with no data at all.
    preform some preprocess on the tile to adjust to common slides issue (e.g. pen marking) - TODO: Explain
    split the tiles dataset and run classification models.
    experiment with different augmentation, architectures, hyperparams as much as possible.

    Some of the described steps were implemented naively due to time constraints.

### Data & More:
    I used data from TUPAC16 change [here](https://tupac.grand-challenge.org/TUPAC/)
    I also used and influenced from other github repositories I found working on that challenge in previous year [deep-histopath](https://github.com/CODAIT/deep-histopath/tree/master)

### Detailed Steps:
    1. **slide to tiles**: using openslide with some github repo i found, I split the slide and,........
    2. **preprocess**: ......
    3. **training**: class imbalance in the slides, best way to address is to actually sample evenly for both classes. because cell are spatially related, I did not see a big advantage in Transformer architectures and experimented mainly with Resnet
    4. 


## Results:

### Performance evaluation
    The Data is a bit tricky, as the image is too big to feed in the model as is.
    I choose to split the image to predefined tiles (crops) and run the training on each tile,
    This means each tile inherits it's Slide tag (In Real world that may be meaningless since an image class usually indicates positive sample, but not all tiles of the slide are all positive).
    Using tiles dataset enabled me to have enough data to run train and evaluation to estimate checkpoint performance.
    Note that the train-eval split is not based on slide, meaning all slides will be in both splits
    (to mitigate this issue I add another 3 svs files - ~3GB for a test set)

    All metrics will be evaluated on the tile dataset.

    For the actual goal of classifying Whole-Slide-Image we will use majority voting on all of the tiles for that image

### TODO / Issues
    1. class imbalance - Best option is to sample evenly using the dataloader
 1.  

## Future work
    