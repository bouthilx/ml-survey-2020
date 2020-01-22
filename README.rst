*******************************************************************************************
Survey of machine-learning experimental methods at NeurIPS2019 and ICLR2020 - Data and code
*******************************************************************************************

How do machine-learning researchers run their empirical validation? In
the context of a push for improved reproducibility and benchmarking, this
question is important to develop new tools for model comparison. We ran a
simple survey asking to authors of two leading conferences, NeurIPS 2019
and ICLR 2020, a few quantitative questions on their experimental
procedures.

A `technical report on HAL <https://hal.archives-ouvertes.fr/hal-02447823>`_ summarizes our
finding. It gives a simple picture of how hyper-parameters are set, how
many baselines and datasets are included, or how seeds are used.
Below, we give a very short summary, but please read (and `cite <https://hal.archives-ouvertes.fr/hal-02447823v1/bibtex>`__) 
`the full report <https://hal.archives-ouvertes.fr/hal-02447823>`__ if you are interested.

**Highlights**
The response rates were 35.6% for NeurIPS and 48.6%
for ICLR.
A vast majority of empirical works optimize model hyper-parameters,
thought almost half of these use manual tuning and most of the automatic
hyper-parameter optimization is done with grid search. The typical number
of hyper-parameter set is in interval 3-5, and less than 50 model fits
are used to explore the search space. In addition, most works also
optimized their baselines (typically, around 4 baselines).
Finally, studies typically reported 4 results per model per task to provide a measure of variance, and around 50% of them
used a different random seed for each experiment.

**Sample results**

.. raw:: html

    <img src="https://github.com/bouthilx/ml-survey-2020/raw/master/images/barplot_with_folds_hpo.png" width="400px">

    <p>How many papers with experiments optimized hyperparameters.</p>

    <img src="https://github.com/bouthilx/ml-survey-2020/raw/master/images/barplot_with_folds_hpo_methods.png" width="400px">

    <p>What hyperparameter optimization method were used.</p>

    <img src="https://github.com/bouthilx/ml-survey-2020/raw/master/images/barplot_with_folds_n_datasets.png" width="400px">

    <p>Number of different datasets used for benchmarking.</p>

    <img src="https://github.com/bouthilx/ml-survey-2020/raw/master/images/barplot_with_folds_n_seeds.png" width="400px">

    <p>Number of results reported for each model (ex: for different seeds)</p>

    </br>

These are just samples. Read `the full report <https://hal.archives-ouvertes.fr/hal-02447823>`_ for
more results.

Data
====

The results are saved in 2 csv files: ``data/iclr.csv`` and ``data/neurips.csv``.

The function ``load_data_pd(path)`` in file ``main.py`` gives an example on how to load the data with ``panda``.

Requirements
============

There is no strict requirement to work with the csv files. To use the code in `main.py` or `barplot.py`, 
you can install the requirements using this command.

::

    $ pip install -r requirements.txt

Generating plots
================

You can generate the plots from report using the script ``main.py``. 

::

    $ python main.py --help
    usage: main.py [-h] [--output OUTPUT] [--type {pdf,png}]
                   [--plot [{emp-vs-theory,hpo,hpo-methods,n-hps,n-hps-grid-search,n-trials,n-trials-grid-search,baseline,n-baselines,n-datasets,n-seeds,seeding,n-points} [{emp-vs-theory,hpo,hpo-methods,n-hps,n-hps-grid-search,n-trials,n-trials-grid-search,baseline,n-baselines,n-datasets,n-seeds,seeding,n-points} ...]]]
    
    optional arguments:
      -h, --help            show this help message and exit
      --output OUTPUT
      --type {pdf,png}
      --plot [{emp-vs-theory,hpo,hpo-methods,n-hps,n-hps-grid-search,n-trials,n-trials-grid-search,baseline,n-baselines,n-datasets,n-seeds,seeding,n-points} [{emp-vs-theory,hpo,hpo-methods,n-hps,n-hps-grid-search,n-trials,n-trials-grid-search,baseline,n-baselines,n-datasets,n-seeds,seeding,n-points} ...]]

Example to generate the plot for question 3, 4, 5 and 6 in pdf format, in folder img.

::

    $ python main.py --output img --type pdf --plot hpo hpo-methods n-hps n-nps-trials


Citation
========

If this report is helpful for your research, please cite it using the following bibtex entry.

.. code-block:: bibtex

  @techreport{bouthillier:hal-02447823,
    TITLE = {{Survey of machine-learning experimental methods at NeurIPS2019 and ICLR2020}},
    AUTHOR = {Bouthillier, Xavier and Varoquaux, Ga{\"e}l},
    URL = {https://hal.archives-ouvertes.fr/hal-02447823},
    TYPE = {Research Report},
    INSTITUTION = {{Inria Saclay Ile de France}},
    YEAR = {2020},
    MONTH = Jan,
    PDF = {https://hal.archives-ouvertes.fr/hal-02447823/file/ml_methods_survey.pdf},
    HAL_ID = {hal-02447823},
    HAL_VERSION = {v1},
  }
