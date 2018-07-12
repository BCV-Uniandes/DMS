# dmn-pytorch
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![codecov](https://codecov.io/gh/andfoy/query-objseg/branch/master/graph/badge.svg?token=99Q4tadxnT)](https://codecov.io/gh/andfoy/query-objseg)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/992bf5adf488489d8ea55998895793c7)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=andfoy/query-objseg&amp;utm_campaign=Badge_Grade)
<!-- [![Build Status](http://157.253.243.11/job/query-objseg/job/master/badge/icon)](http://157.253.243.11/job/query-objseg/job/master/) -->

PyTorch code for Dynamic Multimodal Instance Segmentation guided by natural language queries, ECCV 2018

| ![horses](./examples/horses.png) |
|:--:|
| *A dark horse between three lighter horses* |

<!-- ## Project status -->


## Dependencies

To execute this, you must have Python 3.6.*, [PyTorch](http://pytorch.org/), [Visdom](https://github.com/facebookresearch/visdom), [cupy](http://scikit-image.org/), [Cython](http://cython.org/), [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) installed. To accomplish this, we recommend installing the [Anaconda](https://www.anaconda.com/download) Python distribution and use conda to install the dependencies, as it follows:

```bash
conda install matplotlib numpy cython
conda install pytorch torchvision cuda90 -c pytorch
conda install aria2 -c bioconda
pip install -U visdom opencv-python cupy-cuda90 pynvrtc tqdm
```

You will also require the ReferIt loader library, which you can clone from: https://github.com/andfoy/refer. To install it, you can use ``pip`` as it follows:

```bash
pip install git+https://github.com/andfoy/refer.git
```

Finally, you will need to install the Simple Recurrent Unit (SRU):
```bash
pip install git+https://github.com/taolei87/sru.git
```
Conda packages will be created on future releases.

## Dataset download

Additionally, you must download the [ReferIt, UNC, UNC+ and GRef](https://github.com/lichengunc/refer) datasets. To accomplish this, we provide the ``download_dataset.sh`` bash script that will take care of the required downloads.

```bash
bash download_data --path $PATH_TO_STORE_THE_DATASETS
```

### Datasets

| Dataset Name   |      Original Name      |  Splits |
|:----------:|:-------------:|:------:|
| referit |  RefCLEF | train, val, trainval, test |
| unc |    RefCOCO   |   train, val, testA, testB |
| unc+ | RefCOCO+ |   train, val, testA, testB |
| gref | RefCOCOg | train, val |


## Training
To train the model, you will need to provide the path to the directory that contains the aforementioned datasets, as well to other parameters required to train the model. To train the model with the low-resolution setup described on the original paper, please execute:

```bash
python -u -m dmn_pytorch.train --data $PATH_TO_STORE_THE_DATASETS --dataset $DATASET --val $SPLIT_TO_EVALUATE --backend dpn92 --num-filters 10 --lang-layers 3 --mix-we --save-folder $PATH_TO_STORE WEIGHT_SNAPSHOTS --snapshot $PATH_TO_THE_SNAPSHOT_FILE --accum-iters 1
```

To train the model on high-resolution, you just need to add the ``--high-res`` and ``--upsamp-amplification 32`` to the previous command. **Note:** The snapshot file must correspond to the low resolution weights.

To inspect all the available parameters and their description, please execute ``python -m dmn_pytorch.train --help``. Please refer to the datasets table displayed above to get more information about the dataset names and their respective available splits.

## Evaluation
To evaluate the model, you can define the ``--eval-first`` and ``--epochs 0`` parameter flags to ``dmn_pytorch.train`` as it follows:

```bash
python -u -m dmn_pytorch.train --data $PATH_TO_STORE_THE_DATASETS --dataset $DATASET --val $SPLIT_TO_EVALUATE --backend dpn92 --num-filters 10 --lang-layers 3 --mix-we --save-folder $PATH_TO_STORE WEIGHT_SNAPSHOTS --snapshot $PATH_TO_THE_SNAPSHOT_FILE --epochs 0 --eval-first
```

## Performance

<table>
    <thead>
        <tr>
            <th>Layer 1</th>
            <th>Layer 2</th>
            <th>Layer 3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>L1 Name</td>
            <td rowspan=2>L2 Name A</td>
            <td>L3 Name A</td>
        </tr>
        <tr>
            <td>L3 Name B</td>
        </tr>
        <tr>
            <td rowspan=2>L2 Name B</td>
            <td>L3 Name C</td>
        </tr>
        <tr>
            <td>L3 Name D</td>
        </tr>
    </tbody>
</table>
