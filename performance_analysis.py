import numpy as np
import pandas as pd
from Paper.infection_state import computeStates
from Paper.compute_predictors import compute_odds_based_predictors, compute_hazard_based_predictor
import matplotlib.pylab as plt
import seaborn as sns
import csv


def read_steady_states(n, n_vt, m):
    """
    Read simulated steady states
    :param n: Number of types
    :param n_vt: Number of vaccine types
    :param m: interaction structure

    :return: simulated pre- and post-vaccination steady states in a np.array of dimension (n_parameter_sets, n_states)
    """
    pre_filename = '_'.join(m) + '_{}types_0vts_VE0_prevalence.csv'.format(n)
    post_filename = '_'.join(m) + '_{}types_{}vts_VE1_prevalence.csv'.format(n, n_vt)
    pre_df = pd.read_csv(pre_filename, sep=',')
    post_df = pd.read_csv(post_filename, sep=',')
    pre_steady_states = pre_df.values[:, 1:]
    post_steady_states = post_df.values[:, 1:]

    return pre_steady_states, post_steady_states


def compute_performance(n, n_vt, m):
    """
    Compute performance (i.e. proportion of correctly predicted parameter sets)
    of the hazard- and odds-based predictors, i.e. HR_{VT,i}, HR, OR_{VT,i}, OR
    :param n: number of types
    :param n_vt: number of vaccine types
    :param m: interaction structure
    :return: a list of length 4 containing the performance of HR_{VT,i}, HR, OR_{VT,i}, OR, respectively
    """

    # cut-off: cut-off value for the type-specific prevalence
    cut_off = 0.01

    # vts: list of vaccine types
    # nvts: list of non-vaccine types
    # states: list of infection states
    vts = list(range(1, n_vt + 1))
    nvts = list(range(n_vt + 1, n + 1))
    states = computeStates(n)

    # pre_steady_states, post_steady_states: simulated steady states
    # num_simulations: number of parameter sets
    # marginal_operator: left-multiplication with this matrix yields type-specific (marginal) prevalence
    pre_steady_states, post_steady_states = read_steady_states(n, n_vt, m)
    num_simulations = pre_steady_states.shape[0]
    marginal_operator = np.array([[1 if (t + 1) in s.types else 0 for s in states] for t in range(n)])

    pre_type_prevalence = pre_steady_states.dot(marginal_operator.T)
    post_type_prevalence = post_steady_states.dot(marginal_operator.T)
    pre_type_presence = (pre_type_prevalence > cut_off).astype(int)
    post_type_presence = (post_type_prevalence > cut_off).astype(int)
    nvt_operator = np.array([1 if len(set(s.types) & set(nvts)) > 0 else 0 for s in states])
    pre_nvt_prevalence = pre_steady_states.dot(nvt_operator.T)
    post_nvt_prevalence = post_steady_states.dot(nvt_operator)

    overall_outcome = post_nvt_prevalence - pre_nvt_prevalence
    type_outcomes = post_type_prevalence[:, len(vts):] - pre_type_prevalence[:, len(vts):]

    type_or_predictors = np.zeros((num_simulations, len(nvts)))
    type_hr_predictors = np.zeros((num_simulations, len(nvts)))

    correctness_type_or_pred = np.zeros((num_simulations, len(nvts)))
    correctness_type_hr_pred = np.zeros((num_simulations, len(nvts)))

    filename_trans_matrix = '_'.join(m) + '_{}types_0vts_VE0_trans_matrix.csv'.format(n)
    reader = csv.reader(open(filename_trans_matrix), delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    j = 0
    for row in reader:
        trans_matrix = np.reshape(np.array(row), (2 ** n, 2 ** n))

        # Make prediction if at least one vaccine type is present in the pre-vaccination steady state
        if sum(pre_type_presence[j, np.array(vts) - 1]) != 0:
            # compute odds-based predictors
            type_or_predictors[j] = [compute_odds_based_predictors(pre_steady_states[j], states, vts, nvt_) ** pre_type_presence[j, nvt_ - 1] for nvt_ in nvts]
            # compute hazard-based predictors
            type_hr_predictors[j] = [compute_hazard_based_predictor(pre_steady_states[j], trans_matrix, states, vts, nvt_) ** pre_type_presence[j, nvt_ - 1] for nvt_ in nvts]
        j += 1

    overall_or_predictor = np.prod(type_or_predictors, axis=1)
    overall_hr_predictor = np.prod(type_hr_predictors, axis=1)

    correctness_overall_or_pred = ((overall_or_predictor > 1) & (overall_outcome < 0)) | ((overall_or_predictor < 1) & (overall_outcome > 0))
    correctness_overall_or_pred = correctness_overall_or_pred.astype(int)
    correctness_overall_hr_pred = ((overall_hr_predictor > 1) & (overall_outcome < 0)) | ((overall_hr_predictor < 1) & (overall_outcome > 0))
    correctness_overall_hr_pred = correctness_overall_hr_pred.astype(int)

    for nvt in range(len(nvts)):
        correctness_this_type = ((type_or_predictors[:, nvt] > 1) & (type_outcomes[:, nvt] < 0)) | ((type_or_predictors[:, nvt] < 1) & (type_outcomes[:, nvt] > 0))
        correctness_type_or_pred[:, nvt] = correctness_this_type.astype(int)
        correctness_this_type = ((type_hr_predictors[:, nvt] > 1) & (type_outcomes[:, nvt] < 0)) | ((type_hr_predictors[:, nvt] < 1) & (type_outcomes[:, nvt] > 0))
        correctness_type_hr_pred[:, nvt] = correctness_this_type.astype(int)

    # two conditions for including a parameter set
    condition_all_pre_present = np.sum(pre_type_presence, axis=1) == n
    condition_all_vt_post_absent = np.sum(post_type_presence[:, :len(vts)], axis=1) == 0
    conditions = [all(t) for t in zip(condition_all_pre_present, condition_all_vt_post_absent)]

    performance = np.zeros(4)
    per_type_or_performance = np.zeros(len(nvts))
    per_type_hr_performance = np.zeros(len(nvts))

    if sum(conditions) != 0:
        extracted_correctness = np.extract(conditions, correctness_overall_hr_pred)
        performance[1] = np.count_nonzero(extracted_correctness) / sum(conditions)
        extracted_correctness = np.extract(conditions, correctness_overall_or_pred)
        performance[3] = np.count_nonzero(extracted_correctness) / sum(conditions)

        for nvt in range(len(nvts)):
            extracted_correctness = np.extract(conditions, correctness_type_hr_pred[:, nvt])
            per_type_hr_performance[nvt] = np.count_nonzero(extracted_correctness) / sum(conditions)
            extracted_correctness = np.extract(conditions, correctness_type_or_pred[:, nvt])
            per_type_or_performance[nvt] = np.count_nonzero(extracted_correctness) / sum(conditions)

    # compute the type-specific performance by averaging over the performance of all non-vaccine types
    performance[0] = np.mean(per_type_hr_performance)
    performance[2] = np.mean(per_type_or_performance)
    print(performance)

    return performance


if __name__ == '__main__':
    mode = 'acq'
    multi = 'groupwise'
    multi = 'typewise'
    recip = 'structured'
    epsilon = '2'
    model = [mode, multi, recip, epsilon]

    predictors = [r'$HR_{VT,i}$', r'$HR$', r'$OR_{VT,i}$', r'$OR$']
    predictors_names = ['HR_VTi', 'HR', 'OR_VTi', 'OR']
    max_num_types = 4

    all_performance = np.zeros((len(predictors), max_num_types - 1, max_num_types - 1))

    for num_types in range(2, max_num_types + 1):
        for num_vt in range(1, num_types):
            print('Computing performance in a {}-type system with {} vaccine types...'.format(num_types, num_vt))

            all_performance[:, max_num_types - num_vt - 1, num_types - num_vt - 1] = compute_performance(num_types, num_vt, model)

    for i in range(len(predictors)):
        mask = np.zeros((max_num_types - 1, max_num_types - 1))
        mask[np.triu_indices_from(mask, 1)] = True
        plt.figure(figsize=(6, 6))
        with sns.axes_style("white"):
            cm = plt.cm.OrRd
            ax = sns.heatmap(all_performance[i], mask=mask, vmin=.7, vmax=1, linewidths=.5, square=True, cmap=cm,
                             cbar_kws=dict(ticks=np.linspace(0.7, 1, 4)), annot=True, annot_kws={"size": 25},
                             fmt=".2f", cbar=False)
            ax.set_xticklabels(range(1, max_num_types), size=15)
            ax.set_yticklabels(np.arange(max_num_types - 1, 0, -1), size=15)
            ax.set_xlabel('number of non-vaccine types', size=20)
            ax.set_ylabel('number of vaccine types', size=20)
            ax.set_title(predictors[i], size=25)
            filename = '_'.join(model) + '_' + predictors_names[i]
            # plt.savefig(filename + '.png')
            plt.savefig(filename + '.pdf')
            # plt.show()
            plt.close()
