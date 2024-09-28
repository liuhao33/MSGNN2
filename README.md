
# MSGNN

This is the official implementation of MSGNN for Availability Check.

# System Requirements
## Hardware Platform
This code has been tested on the following platform:
+ CPU: I9-14900K 
+ GPU: RTX 4090 
+ Memory: 64GB 

## Software requirements
### OS requirements
This code has been tested on the following systems:

+ Windows 10 (with CUDA 11.6)

### Dependencies

+ python 3.7

+ pytorch 1.12.1

+ scipy 1.7.3

+ networkx 2.6.3

+ scikit-learn 1.0.2

+ dask 2021.10.0


Data preprocessing is partially adopted from [MAGNN](https://github.com/cynricfu/MAGNN).
And dependencies for the preprocessing are not listed here.

# How to use:

### Install dependencies

```shell
conda env create -f env.yaml
conda activate MSGNN
```
or
```shell
conda create -n MSGNN python=3.7
conda activate MSGNN
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install scipy=1.7.3 networkx=2.6.3 scikit-learn=1.0.2 dask=2021.10.0
conda install ipykernel matplotlib

```


### Train and test model on DBLP
The DBLP dataset is adopted from [MAGNN](https://github.com/cynricfu/MAGNN). Based on it, we search schema instances and feed them to our model.
To train the model and test it on link prediction task, run

```shell
python run_DBLP.py
```

### Schema decomposition
The implementation of the proposed decomposition algorithm can be found in `schema_decomposition.ipynb`

### Time consumption of instance search
To exam the time consumption of instance search, open and run `time_DBLP.ipynb`

### Sensitivity analysis
The raw data of sensitivity analysis can be found in `sensitivity.xlsx`

### Generate representations 
To get node representations, run

```shell
python representation_DBLP.py
```
The representations will be saved to `\data\DBLP_processed\representations`
We use [HINormer](https://github.com/Ffffffffire/HINormer) as the classifer on node classifacation task.
