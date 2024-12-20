# NodeSense
This repository provides the implementation for the proposed NodeSense urban sound classifier.
For the peak detection section and demo of the code can be found in the ./recording/ file. 
For the model training we provide the train.py in the root of the repository.

## Model
the model should be put in the root of the folder. downloadable from:
https://drive.google.com/file/d/1JSx-iyr4a5NwXYi3dmYKllkGHRAnN6Ke/view?usp=sharing

By running train.py, the model is fine-tuned after selecting the best values for the hyperparameters. Due to this process taking a long time, we have provided a pre-fine-tuned model for you, which you can download via this link: [[LINK](https://drive.google.com/file/d/1JSx-iyr4a5NwXYi3dmYKllkGHRAnN6Ke/view?usp=sharing)]. Paste this folder in the main folder and run NodeSense via python3 recording/start.py to skip the fine-tuning process

## Peak detection
The peak detection/demo application can be run using the following commands.
```bash
pip install -r ./recording/requirements.txt
python ./recording/start.py
```

## Classifier training
The classifier model can be run using the following commands
```
pip install -r ./requirements.txt
python ./train.py
```

# Authors
- R. Massa (S2848457)
- J. Hebinck (s2736136)
- G. Kleinlein (s3745880)
- J. van den Berg (s2983281)
