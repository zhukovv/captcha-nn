### Requirements

1. Python (tested on 2.7), OpenCV 3
2. Keras, Tensorflow etc: "pip install -r numpy imutils sklearn tensorflow keras"
3. labeled data is expecting to be in "data/labeled_captchas" folder

### Run

Solver:
`python solve_captcha.py`


To retrain the model:
`python train_model.py`

This will write out "captcha_model.hdf5" and "model_labels.dat"

Expects extracted single letters in "data/letter_images" folder

Extract letters: 

`python prepare_data.py`