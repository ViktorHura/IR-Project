# Train emotions classification models
All training scripts will output 
```bash
model.h5
tokenizer.pickle
results.png
stats.txt
checkpoints/model-checkpoint-x.h5
```

in to `results/models` directory,
make sure you move these files into a new folder before training another model, as they will be overwritten!

The validation script generates
```bash
confusion.png
confusion-neutral.png
```
in `results/models/{model-name}` directory.

## Naive model

###Training
```bash

```

###Validation
```bash

```


## SmallerVGG

###Training
```bash
python trainSmallerVGG.py
```
To change batchsize or number of epochs, check the variables at the top of the script.

###Validation
```bash
python valSmallerVGG.py
```
This validates `results/models/smallerVGG-short` and saves to the same directory.

To change the model being validated or output directory, check the variables at the top of the script.

## VGG16

###Training
```bash

```

###Validation
```bash

```

# Classifying Lexica data

## using VGG16
```bash

```

## using BERT
```bash

```