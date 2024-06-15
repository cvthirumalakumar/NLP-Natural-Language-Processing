# NLP Assignment 3

Foobar is a Python library for dealing with word pluralization.

## Implementation
1. All the .py files has full steps for training word vectors and classification repectively. 
2. Input csv file locations and output pt file location can be changed accordingly.
3. All words having frequency <=3 are replaced with unk token.
4. Stemming, expansion of short words to full forms (eg. don't => do not), numbers are replaced with num tag, all punctuations are removed as pre-processing steps.
4. svd and skipgram is trained using entire corpus and word embedding size considered is 256.
5. Both svd and skipgrams are trained for window sizes 1,2,3,4, 5 and trained downstream classification models with all 5 different sizes metioned earlier.

## Pretrained models
Pretrained modela are avilable in the [Google drive link](https://drive.google.com/drive/folders/16XMvWbSkfWo7uYCLUDfXBG73WDyYBQ72?usp=sharing)

Loading svd vectors
```python
word2idx,svd_word_vectors =  torch.load('svd-word-vectors.pt')
```
same word2idx is used for both svd and skipgram and word2idx is saved only in `svd-word-vectors.pt` file.

Loading skipgram vectors
 ```python
skipgram_word_vectors =  torch.load('skipgram-word-vector.pt', map_location='cpu')
```
`skipgram_word_vectors` variable is a dictionary and has 2 embeddings for context word and target word. Both of these can be accessed using keys `context_embeddings` and `target_embeddings` respectively. Summation of both is considered when training downstream models.

Skipgram is trained for 5 epochs.

## Downstream Classification Training
1. Train and Validation dataset is randomly splitted in 80:20 proportion from train set.
2. All models trained using Early stopping criteria based on validation loss with patience=5 and number of epochs given is 50.
3. All other detals can be found in the code.


