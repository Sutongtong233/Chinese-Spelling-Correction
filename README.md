# Chinese-Spelling-Correction
NLP2021 Course Project

## Requirements

To install requirements:

```setup
pip install bert4keras
```

## Pre-trained Models

The pre-trained **Chinese Bert** has been downloaded at ./chinese_L-12_H-768_A-12.

## Dataset

At ./data. Train, Validation and test data has been splitted, in json form.

## Training

We provide 2 versions. Correlation_basic and Correlation_mlm.

### Correlation_basic

Directly using pre-trained model and calculating character similarity. The weight of three kinds of metrics of similarity are hyperparameters.
```train
python correlation_basic.py 
```

### Correlation_mlm

Use train_json to do fine tuning. 

```train
python correlation_mlm.py
```


## Case Study Evaluation

```eval
if __name__ == '__main__':
    text = '专家公步虎门大桥涡振原因'
    result = text_correction(text)
    print(result)
```


