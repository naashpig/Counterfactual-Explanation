# Introduction on counterfactual explanation via prototype learning
This repository is the implementation of ["Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"](https://dl.acm.org/doi/pdf/10.1145/3351095.3372850) and ["Interpretable Counterfactual Explanations Guided by Prototypes"](https://arxiv.org/pdf/1907.02584.pdf)
## Requirements
To install requirements:

```sh
conda env create -f env.yml
conda activate CFE
```

## Getting start with coutnerfactual explanation (CFE) 
To run the model and get the counterfacual instance for the input, run this command with :
```
python test.py
```
## Choosing the test sample
It is possible to change the values of the test sample with any value you want
```py
test_sample = {'age':34,
                  'workclass':'Private',
                  'education':'Bachelors',
                  'marital_status':'Single',
                  'occupation':'Service',
                  'race': 'White',
                  'gender':'Male',
                  'hours_per_week': 45}
```
## Prototype Learning
We propose using class prototypes in the objective function to guide the perturbations quickly towards an interpretable counterfactual. Similar with the [referred paper](https://arxiv.org/pdf/1907.02584.pdf), We select the target class prototype as the average over the K nearest train instances with the counterfactual class label not in the latent space. When we decide to use this learning method, it is necessary to choose K.

```py
PrototypeMode=True # or False
num_prototype_k=10
```

## Acknowledgement
This code is adapted and modified upon the code [github](https://github.com/interpretml/DiCE) of FAT 2020 paper "Explaining Machine Learning Classifiers through Diverse
Counterfactual Explanations". We appreciate their released dataset and code which are very helpful to our research.
