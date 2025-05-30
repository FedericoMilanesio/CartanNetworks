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

## Training

To train the classification models from Fig. 2, run:

```
python code/experiments/run_classification.py
```

To train the regression models, run

```
python code/experiments/run_regression.py
```

for Tab 3. experiments and

```
python code/experiments/run_kmnist.py
```

and

```
python code/experiments/run_cifar.py
```

for Fig 3. expriments.

## Evaluation

To evaluate our models, follow the notebook 'statistical_tests/testing.ipynb'.