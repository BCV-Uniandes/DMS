# query-objseg
Semantic Segmentation based on Natural Language Queries (WIP)

## Project status
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## Dependencies

To execute this, you must have Python 3.6.*, [PyTorch](http://pytorch.org/), [Visdom](https://github.com/facebookresearch/visdom), [scikit-image](http://scikit-image.org/), [Cython](http://cython.org/), [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) installed. To accomplish this, we recommend installing the [Anaconda](https://www.anaconda.com/download) Python distribution and use conda to install the dependencies, as it follows:

```bash
conda install matplotlib numpy cython scikit-image
conda install pytorch torchvision cuda80 -c soumith
conda install aria2 -c bioconda
pip install -U visdom
```

You will also require ReferIt loader library, which you can clone from: https://github.com/andfoy/refer. To install it, you can use ``pip`` as it follows:

```bash
cd <path-to-the-library-clone>
pip install .
```

Conda packages will be created on future releases.

## Dataset download
You must download the [ReferIt](https://github.com/lichengunc/refer) dataset, as well the train/val/test split used for our experiments. For this, we provide the ``download_dataset.sh`` bash script that will take care of the downloads required.
