# NodeSense
This repository provides the implementation for the proposed NodeSense urban sound classifier.
For the peak detection section and demo of the code can be found in the ./recording/ file. 
For the model training we provide the trian.py in the root of the repository.

## Peak detection
The peak detection/demo application can be run using the following commands.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ./recording/requirements.txt
python ./recording/start.py
```

## Classifier training
The classifier model can be run using the following commands
```
python3 -m venv venv
source venv/bin/activate
pip install -r ./requirements.txt
python ./train.py
```

## Cross-fold validation
The cross-fold validation of the model can be run using the following commands
```
python3 -m venv venv
source venv/bin/activate
pip install -r ./requirements.txt
python ./crossfold.py
```

# Authors
- R. Massa (S2848457)
- J. Hebinck (s2736136)
- G. Kleinlein (s3745880)
- J. van den Berg (s2983281)
