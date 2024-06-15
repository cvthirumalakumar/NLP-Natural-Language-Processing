# NLP Assignment 4


## Implementation
1. All the .py files has full steps for training word vectors and classification repectively. 
2. Input csv file locations and output pt file location can be changed accordingly.
3. All words having frequency <=3 are replaced with unk token.
4. Stemming, expansion of short words to full forms (eg. don't => do not), numbers are replaced with num tag, all punctuations are removed as pre-processing steps.
4. word embedding size considered is 256.

## Pretrained models
Pretrained modela are avilable in the [Google drive link](https://drive.google.com/drive/folders/1S_DFtlyazBG79ipcdiQiDUyX02Ix-8nn?usp=sharing)
ELMO vocab is saved as `elmo_vocab.pkl`

## Downstream Classification Training
1. Train and Validation dataset is randomly splitted in 80:20 proportion from train set.
2. All models trained using Early stopping criteria based on validation loss with patience=5 and number of epochs given is 25.
3. All other detals can be found in the code.


