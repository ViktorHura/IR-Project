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

### Training
```bash
python trainNaive.py
```
To change batchsize or number of epochs, check the variables at the top of the script.

### Validation
```bash
python valNaive.py
```
This validates `results/models/naive-short` and saves to the same directory.

To change the model being validated or output directory, check the variables at the top of the script.


## SmallerVGG

### Training
```bash
python trainSmallerVGG.py
```
To change batchsize or number of epochs, check the variables at the top of the script.

### Validation
```bash
python valSmallerVGG.py
```
This validates `results/models/smallerVGG-short` and saves to the same directory.

To change the model being validated or output directory, check the variables at the top of the script.

## VGG16

### Training
```bash
python trainVGG16.py
```
To change batchsize or number of epochs, check the variables at the top of the script.

### Validation
```bash
python valVGG16.py
```
This validates `results/models/VGG16-short` and saves to the same directory.

To change the model being validated or output directory, check the variables at the top of the script.

## BERT

### Validation
```bash
python valBERT.py
```
This validates a pretrained BERT model and saves to `results/models/BERT`.

# Classifying Lexica data
These scripts will take the preprocessed Lexica data from `results/data/cleanData.csv` and label all 1.5M entries, resulting in a new csv.

These scripts take a couple of hours for the entire dataset, make sure pytorch and tensorflow have gpu acceleration enabled!
The VGG16 model is much faster to inference.

## using VGG16
```bash
python VGG16Label.py
```
Saved in `results/data/VGG16-labelled.csv`

## using BERT
```bash
python BERTLabel.py
```
Saved in `results/data/BERT-labelled.csv`