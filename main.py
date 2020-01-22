import argparse
from collections import OrderedDict
import copy
import os

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np
import pandas as pd
from scipy.stats import t

from barplot import barplot_with_folds, barplot_vertical

plt.rcParams.update({'font.size': 8})
plt.close('all')
plt.rcParams["font.family"] = "Times New Roman"
# Fix on ubuntu, otherwise Times New Roman renders in bold
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)

COLORS = """\
#46a285
#dc6d42
#6d80ab
#c76aa3
#86a824
#dfb90e""".split('\n')

NOT_APPLICABLE_COLOR = COLORS.pop(-1)


def load_data_pd(path):

    data = pd.read_csv(path)

    column_names = {
        'Timestamp': 'timestamp',
        ('Did you have any experiments in your paper? If no, you are already done with '
         'the survey, thank you and have a good day. :)'): 'emp_or_theory',
        'Did you optimize your hyperparameters?': 'hpo',
        'If yes, how did you tune them?': 'hpo_method',
        'How many hyperparameters did you optimize?': 'n_hps',
        ('How many trials/experiments in total during the optimization? (How many '
         'different set of hyperparameters were evaluated)'): 'n_trials',
        ('Did you optimize the hyperparameters of your baselines? (The other models or '
         'algorithms you compared with)'): 'baseline',
        ('How many baselines (models, algos) did you compare with? (If different '
         'across datasets, please report maximum number, not total)'): 'n_baselines',
        'How many datasets or tasks did you compare on?': 'n_datasets',
        'How many results did you report for each model (ex: for different seeds).': 'n_seeds',
        ('If you answered 1 to previous question, did you use the same seed in all '
         'your experiments? (If you did not seed your experiments, you should answer '
         '\'no\' to this question.)'): 'seeding'
    }

    data.rename(columns=column_names, inplace=True)

    return data


def compute_ci(cs, alpha=0.05):
    total = cs.sum()
    p = cs / total
    var = total * p * (1 - p)    
    se = np.sqrt(var) 
    t_value = t.ppf(1 - alpha, df=total  - 1)
    return t_value * se
 
 
def compute_p_ci(cs, alpha=0.05):
    ci = compute_ci(cs, alpha)
    return ci / cs.sum() * 100
 

def generate_counts(data, labels, group, order, count_not_applicable=False, get_subset=None):

    plot_data = {}

    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        if get_subset:
            conf_data = conf_data[get_subset(conf_data)]

        group_data = conf_data.groupby(group).size()

        plot_data[conf_name] = [group_data.get(key, 0) for key in order]

        if count_not_applicable:
            not_applicable = len(conf_data[group]) - sum(plot_data[conf_name])
            plot_data[conf_name].insert(0, not_applicable)

    return plot_data


def infer_points(data, hpo_method='Grid Search'):

    n_hps = ['1', '2', '3-5', 'More than 5']
    n_trials = ['1-50', '50-100', '100-200', '200-500', 'More than 500']

    convert_n_hps = {
        '1': 1,
        '2': 2,
        '3-5': 3,
        'More than 5': 6}

    convert_n_trials = {
        '1-50': 50,
        '50-100': 100,
        '100-200': 200,
        '200-500': 500,
        'More than 500': 1000}

    groups = dict()
    counts = dict()

    all_hps = data.groupby('n_trials').size()
    total = float(all_hps.sum())

    for n_hp in n_hps:
        group = data[data['n_hps'] == n_hp].groupby('n_trials').size()
        for label in n_trials:
            count = group.get(label, 0)
            n_points = int(np.exp(np.log(convert_n_trials[label]) /
                                  float(convert_n_hps[n_hp])) + 0.5)
            if n_points not in counts and count:
                counts[n_points] = 0
            
            if count:
                counts[n_points] += count

    total = len(data)

    labels = sorted(counts.keys())
    values = [counts[label] for label in labels]

    assert total == sum(values)

    cum = ((np.array(values) / total).cumsum() < 0.5).sum() + 0.5

    return labels, values, cum


def barplot_with_folds_emp_vs_theory(output, labels, data, img_type):

    order = ['Yes', 'No'][::-1]
    names = ['Theoretical', 'Empirical']
    
    plot_data = generate_counts(data, labels, 'emp_or_theory', order)

    barplot_with_folds(labels, names, plot_data, COLORS,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_emp_or_theory.{img_type}'))


def barplot_with_folds_hpo(output, labels, data, img_type):

    order = ['Yes', 'No'][::-1]
    names = ['With optimization', 'Without optimization', 'Not applicable'][::-1]
    
    plot_data = generate_counts(
        data, labels, 'hpo', order, count_not_applicable=True,
        get_subset=lambda conf_data: conf_data['emp_or_theory'] == 'Yes')

    colors = copy.deepcopy(COLORS)
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_hpo.{img_type}'))


def barplot_with_folds_hpo_methods(output, labels, data, img_type):

    plot_data = {}

    names = ['Manual Tuning', 'Grid Search', 'Random Search', 'Other'][::-1]
    
    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        conf_data = conf_data[(conf_data['emp_or_theory'] == 'Yes') * (conf_data['hpo'] == 'Yes')]
        counts = [conf_data['hpo_method'].str.contains(hpo_method).sum()
                  for hpo_method in names]

        plot_data[conf_name] = counts

    barplot_with_folds(labels, names, plot_data, COLORS,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_hpo_methods.{img_type}'))


def barplot_with_folds_n_hps(output, labels, data, img_type):
    order = ['1', '2', '3-5', 'More than 5'][::-1]
    names = ['1 hyperparameter', '2 hyperparameters', '3-5 hyperparameters',
             '5+ hyperparameters', 'Not applicable'][::-1]

    def get_subset(conf_data):
        return (conf_data['emp_or_theory'] == 'Yes') * (conf_data['hpo'] == 'Yes')

    plot_data = generate_counts(
        data, labels, 'n_hps', order, count_not_applicable=True, get_subset=get_subset)
    
    colors = copy.deepcopy(COLORS)[:4]
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_n_hps.{img_type}'))


def barplot_with_folds_n_hps_hpo_method(output, labels, data, img_type, hpo_method='Grid Search'):
    order = ['1', '2', '3-5', 'More than 5'][::-1]
    names = ['1 hyperparameter', '2 hyperparameters', '3-5 hyperparameters',
             '5+ hyperparameters'][::-1]

    def get_subset_hpo_method(conf_data):
        return ((conf_data['emp_or_theory'] == 'Yes') * 
                (conf_data['hpo'] == 'Yes') * 
                (conf_data['hpo_method'].str.contains(hpo_method, na=False)))

    plot_data = generate_counts(
        data, labels, 'n_hps', order, count_not_applicable=False, get_subset=get_subset_hpo_method)

    filename = 'barplot_with_folds_n_hps_{hpo_method}.{img_type}'.format(
        hpo_method=hpo_method.replace(' ', '_'), img_type=img_type)

    barplot_with_folds(labels, names, plot_data, COLORS,
                       height=1.6, width=4,
                       filename=os.path.join(output, filename))


def barplot_with_folds_n_trials(output, labels, data, img_type):

    order = ['1-50', '50-100', '100-200', '200-500', 'More than 500'][::-1]
    names = ['1-50 trials', '50-100 trials', '100-200 trials', '200-500 trials',
             '500+ trials', 'Not applicable'][::-1]

    def get_subset(conf_data):
        return (conf_data['emp_or_theory'] == 'Yes') * (conf_data['hpo'] == 'Yes')

    plot_data = generate_counts(
        data, labels, 'n_trials', order, count_not_applicable=True, get_subset=get_subset)

    colors = copy.deepcopy(COLORS)
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_n_trials.{img_type}'))

def barplot_with_folds_n_trials_hpo_method(output, labels, data, img_type, hpo_method='Grid Search'):

    order = ['1-50', '50-100', '100-200', '200-500', 'More than 500'][::-1]
    names = ['1-50 trials', '50-100 trials', '100-200 trials', '200-500 trials',
             '500+ trials'][::-1]

    def get_subset_hpo_method(conf_data):
        return ((conf_data['emp_or_theory'] == 'Yes') * 
                (conf_data['hpo'] == 'Yes') * 
                (conf_data['hpo_method'].str.contains(hpo_method, na=False)))

    plot_data = generate_counts(
        data, labels, 'n_trials', order, count_not_applicable=False, get_subset=get_subset_hpo_method)

    filename = 'barplot_with_folds_n_trials_{hpo_method}.{img_type}'.format(
        hpo_method=hpo_method.replace(' ', '_'), img_type=img_type)

    barplot_with_folds(labels, names, plot_data, COLORS,
                       height=1.6, width=4,
                       filename=os.path.join(output, filename))


def barplot_with_folds_baseline(output, labels, data, img_type):

    plot_data = {}

    group = 'baseline'

    order = ['Yes', 'No', 'Not applicable'][::-1]
    names = ['With optimization', 'Without optimization', 'Not applicable'][::-1]
    
    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        conf_data = conf_data[(conf_data['emp_or_theory'] == 'Yes')]

        group_data = conf_data.groupby(group).size()

        plot_data[conf_name] = [group_data[key] for key in order]
        plot_data[conf_name][0] += len(conf_data[group]) - sum(plot_data[conf_name])

    colors = copy.deepcopy(COLORS)[:2]
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_baseline.{img_type}'))


def barplot_with_folds_n_baselines(output, labels, data, img_type):

    plot_data = {}

    group = 'n_baselines'

    order = ['1', '2', '3-5', '5-10', 'More than 10'][::-1]
    names = ['1 baseline', '2 baselines', '3-5 baselines', '5-10 baselines',
             '10+ baselines', 'Not applicable'][::-1]
    
    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        conf_data = conf_data[(conf_data['emp_or_theory'] == 'Yes')]

        group_data = conf_data.groupby(group).size()

        plot_data[conf_name] = [group_data[key] for key in order]

        plot_data[conf_name] = [group_data[key] for key in order]
        plot_data[conf_name].insert(0, len(conf_data[group]) - sum(plot_data[conf_name]))

    colors = copy.deepcopy(COLORS)
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_n_baselines.{img_type}'))


def barplot_with_folds_n_datasets(output, labels, data, img_type):

    plot_data = {}

    group = 'n_datasets'

    order = ['1', '2', '3-5', '5-10', 'More than 10'][::-1]
    names = ['1 dataset', '2 datasets', '3-5 datasets', '5-10 datasets',
             '10+ datasets', 'Not applicable'][::-1]
    
    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        conf_data = conf_data[(conf_data['emp_or_theory'] == 'Yes')]

        group_data = conf_data.groupby(group).size()

        plot_data[conf_name] = [group_data[key] for key in order]

        plot_data[conf_name] = [group_data[key] for key in order]
        plot_data[conf_name].insert(0, len(conf_data[group]) - sum(plot_data[conf_name]))

    colors = copy.deepcopy(COLORS)
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_n_datasets.{img_type}'))



def barplot_with_folds_n_seeds(output, labels, data, img_type):

    plot_data = {}

    group = 'n_seeds'

    order = ['1', '2', '3-5', '5-10', 'More than 10'][::-1]
    names = ['1 sample', '2 samples', '3-5 samples', '5-10 samples',
             '10+ samples', 'Not applicable'][::-1]
    
    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        conf_data = conf_data[(conf_data['emp_or_theory'] == 'Yes')]

        group_data = conf_data.groupby(group).size()

        plot_data[conf_name] = [group_data[key] for key in order]

        plot_data[conf_name] = [group_data[key] for key in order]
        plot_data[conf_name].insert(0, len(conf_data[group]) - sum(plot_data[conf_name]))

    colors = copy.deepcopy(COLORS)
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_n_seeds.{img_type}'))


def barplot_with_folds_seeding(output, labels, data, img_type):

    plot_data = {}

    group = 'seeding'

    order = ['Yes, all on the same seed',
            'No, I did not seed my experiments or I explicitly used different seeds']
    names = ['Not applicable', 'Different seeds or\nnon seeded', 'Same seed']

    yes = 'Yes, all on the same seed'
    no = 'No, I did not seed my experiments or I explicitly used different seeds'
    corrections = {
        ('I reported average score for each model, averaged over 10 runs. Each of the '
         '10 runs used the same seed across all models. '): yes,
        ('Same seed, but estimated the variance in previous works to be small.'): no,
        ('not applicable. there\'s no seed because we use pre-trained models'): no,
        ('Seeded the data examples, did not seed the simulations (where however enough '
         'Monte Carlo replicates were conducted to make noise negligible)'): yes,
        ('Our paper did not involve training deep networks, and there were not any '
         'mysterious hyperparameters....'): yes,  # because the model was deterministic it seems.
        ('Checked that results were consistent with different seeds but no proper '
         'study of variance'): no,  # because they tried different seeds but only reported one
        'Choice of seed has no effect on training outcome': yes  # Assuming they are right.
        }
    
    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        conf_data = conf_data[(conf_data['emp_or_theory'] == 'Yes') * (conf_data['n_seeds'] == '1')]

        for key, new_value in corrections.items():
            conf_data.loc[conf_data['seeding'] == key, 'seeding'] = new_value

        group_data = conf_data.groupby(group).size()

        plot_data[conf_name] = [group_data[key] for key in order]
        plot_data[conf_name].insert(0, len(conf_data['seeding']) - sum(plot_data[conf_name]))

    colors = copy.deepcopy(COLORS)
    colors.insert(0, NOT_APPLICABLE_COLOR)

    barplot_with_folds(labels, names, plot_data, colors,
                       height=1.6, width=4,
                       filename=os.path.join(output, f'barplot_with_folds_seeding.{img_type}'))


def barplot_n_points(output, labels, data, img_type, hpo_method='Grid Search'):
    for i, conf_name in enumerate(labels):
        conf_data = data[conf_name]
        conf_data = (conf_data[(conf_data['emp_or_theory'] == 'Yes') * (conf_data['hpo'] == 'Yes') *
                     conf_data['hpo_method'].str.contains(hpo_method, na=False)])

        labels, values, threshold = infer_points(conf_data, hpo_method=hpo_method)

        if conf_name == 'NeurIPS':
            height = 11 / 4
            width = (8.5 - 2) * 0.45
        else:
            height = 11 / 4 * (12/14.)
            width = (8.5 - 2) * 0.418

        if conf_name == 'NeurIPS':
            vmax = 60
        else:
            vmax = 50

        xlabel = 'Number of papers'
        if conf_name == 'NeurIPS':
            ylabel = 'Number of values\nper dimension'
        else:
            ylabel = None

        filename = 'barplot_{conf}_n_points.{img_type}'.format(conf=conf_name, img_type=img_type)
        filename = os.path.join(output, filename)
        barplot_vertical(labels, values, filename, vmax, width, height, xlabel, ylabel=ylabel,
                         threshold=threshold)


plots = {
    'emp-vs-theory': barplot_with_folds_emp_vs_theory,
    'hpo': barplot_with_folds_hpo,
    'hpo-methods': barplot_with_folds_hpo_methods,
    'n-hps': barplot_with_folds_n_hps,
    'n-hps-grid-search': barplot_with_folds_n_hps_hpo_method,
    'n-trials': barplot_with_folds_n_trials,
    'n-trials-grid-search': barplot_with_folds_n_trials_hpo_method,
    'baseline': barplot_with_folds_baseline,
    'n-baselines': barplot_with_folds_n_baselines,
    'n-datasets': barplot_with_folds_n_datasets,
    'n-seeds': barplot_with_folds_n_seeds,
    'seeding': barplot_with_folds_seeding,
    'n-points': barplot_n_points}


def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--output', default='images')
    parser.add_argument('--type', default='png', choices=['pdf', 'png'])
    parser.add_argument('--plot', nargs='*', default=plots.keys(), choices=plots.keys())

    options = parser.parse_args(argv)

    conf = ['neurips', 'iclr']
    labels = {'neurips': 'NeurIPS', 'iclr': 'ICLR'}

    data = {labels[name]: load_data_pd('data/{name}.csv'.format(name=name)) for name in conf}

    labels = ['NeurIPS', 'ICLR']

    if not os.path.isdir(options.output):
        os.mkdir(options.output)

    for plot in options.plot:
        print(plot)
        plots[plot](options.output, labels, data, options.type)


if __name__ == '__main__':
    main()
