# Cartan Networks: Group theoretical Hyperbolic Deep Learning

This repository is the official implementation of [Cartan Networks: Group theoretical Hyperbolic Deep Learning](https://arxiv.org/). 

## Requirements

To create and activate the Conda environment from the provided `.yml` file:

1. **Create the environment**:

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**:

   ```bash
   conda activate hyperbolic
   ```

3. **Pull submodules**:
   ```bash
   git submodule update --init --recursive
   ``` 

## Training

To train the feedforward classification models, run:

```
python code/experiments/run_classification.py
```

To train the convolutional networks, run first:
```
python prepare_imagenet.py
```
and then either
```
python code/experiments/run_alexnet.py
```
or 
```
python code/experiments/run_resnet.py
```
Due to google drive restrictions, you will likely have to download the CelebA dataset on your own.
## Evaluation

To evaluate our models, follow the notebook 'statistical_tests/testing.ipynb'.
