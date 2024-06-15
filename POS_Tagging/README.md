
# Instructions for running codes
## Training
All training, testing and graphs generation codes are in `INLP_Assignment_2_lstm_ffnn_training_and_testing.ipynb` notebook.

## Inferencing
Pretrained models are available and can be downlaoded form this [drive line](https://drive.google.com/drive/folders/1WJk_LQbu8PyNzm-h7on0c558vPlWaXlE?usp=sharing). Place these models in the same directory before running inference code.

Please run the following code to inference the models.
```
python pos_tagger.py -f
```
Use the argument `-f` for using FFNN based pos tagger and `-r` for using LSTM based pos tagger.

Upon running the above `.py` file, it will promt for a sentence and will prints the corresponding word and pos tag pairs.

Please keep model files (.pt) and vocab files (.pkl) in the same directory.
