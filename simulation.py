import numpy as np
from time import time, strftime
import pandas as pd
import Paper.parameters_generation
from Paper.ode_model import get_steady_state
from Paper.infection_state import computeStates, getState
import csv


def read_parameters(n):
    """
    Read generated parameter sets with n_types.
    :param n: number of types
    :return: np.array of dimension (num_simulations, n_types + 2 * n_types**2)
        - the first n_types columns: the type-specific transmissibility
            = (contact rate * probability of successful acquisition given contact)
        - the next n_types**2 columns: give the interaction parameters in acquisition
        - the last n_types**2 columns: give the interaction parameters in clearance
    """
    filename = 'parameters_{}_types.csv'.format(n)
    df = pd.read_csv(filename, sep=',')
    output = df.values[:, 1:]
    return output


def read_partitions(n):
    """
    Read sets of partitions of clusters(groups) with n_types.
    :param n: Number of types
    :return: np.array of dimension: (num_simulations, n_types)
    """
    filename = 'partitions_{}_types.csv'.format(n)
    df = pd.read_csv(filename, sep=',')
    output = df.values[:, 1:]   # The 0-th column of df consists of useless indices.
    return output


def read_sign(n):
    """
    Read sets of generated parameters with n_types.
    :param n: Number of types
    :return: np.array of dimension (num_simulations, n_types + n_types**2)
    """
    # filename = folder + 'parameters_{}_types.csv'.format(n_types)
    filename = 'sign_parameters_{}_types.csv'.format(n)
    df = pd.read_csv(filename, sep=',')
    output = df.values[:, 1:]   # The 0-th column of df consists of useless indices.
    return output


def derive_param(parameter_set, sign, untransformed_partition, n, n_vt, m):
    """
    :param parameter_set: transmissibilities and interaction parameters
    :param sign: signs parameters
    :param untransformed_partition: partition of groups (clusters) of types in an untransformed format
    :param n: number of types
    :param n_vt: number of vaccine types
    :param m: mode = [mode, multi, recip, epsilon]
    :return:
    """
    k_matrix, h_matrix, partition = [], [], []
    states = computeStates(n - n_vt)

    if m[1] == 'typewise':
        # typewise multiplicative structure
        partition = []
        # Derive unstructured (asymmetric) matrices for interaction in acquisition and clearance
        k_matrix = np.reshape(parameter_set[n:int(n + n ** 2)], (n, n))[n_vt:, n_vt:]
        h_matrix = np.reshape(parameter_set[int(n + n ** 2):], (n, n))[n_vt:, n_vt:]

        if m[2] == 'structured':
            # Get the ratio between reciprocal pairwise interaction parameters
            eps = int(m[3]) / 10
            k_matrix = np.triu(k_matrix)
            h_matrix = np.triu(h_matrix)

            # Derive structured (symmetric) matrices for interaction in acquisition and clearance
            k_matrix += np.triu(k_matrix, 1).T
            h_matrix += np.triu(h_matrix, 1).T

            # Pull away reciprocal pairwise interaction parameters into ratios of eps
            k_matrix = k_matrix * np.exp(sign[:int(n ** 2)].reshape((n, n)) * eps / 2)[n_vt:, n_vt:]
            h_matrix = h_matrix * np.exp(sign[int(n ** 2):].reshape((n, n)) * eps / 2)[n_vt:, n_vt:]
    else:
        # groupwise multiplicative structure
        assignment = np.array(untransformed_partition)
        n_clus = len(np.unique(assignment))
        reduced_assignment = assignment[n_vt:].astype(int)
        reduced_clusters = np.unique(reduced_assignment)
        partition = [list(np.where(reduced_assignment == c)[0] + 1) for c in reduced_clusters]
        states = computeStates(n - n_vt, partition)

        k_matrix = np.triu(np.reshape(parameter_set[n:(n + n_clus ** 2)], (n_clus, n_clus)))
        h_matrix = np.triu(np.reshape(parameter_set[(n + n_clus ** 2):(n + 2 * n_clus ** 2)], (n_clus, n_clus)))

        k_matrix += np.triu(k_matrix, 1).T
        h_matrix += np.triu(h_matrix, 1).T

        k_matrix = np.take(k_matrix[reduced_clusters - 1], reduced_clusters - 1, 1)
        h_matrix = np.take(h_matrix[reduced_clusters - 1], reduced_clusters - 1, 1)

    if m[0] == 'acq':
        h_matrix = np.ones(h_matrix.shape)

    return k_matrix, h_matrix, states, partition


def simulate_pre_vaccination(n, m):
    # """
    # :param n: number of types
    # :param m: interaction structure
    # """

    # param parameter_sets: np.array of parameter sets (transmissibilities and interaction parameters)
    # part_sets: np.array of partition (clusters/groups of types) sets
    # sign_sets: np.array of sign parameters
    # num_simulations: number parameter sets to simulate from
    # states: list of infection states
    # mu: the baseline per capita type-specific clearance rate
    # marginal operator: matrix for left-multiplying with the state variable to get the marginal prevalence
    # steady_states: np.array for saving the simulated steady state prevalence per infection state
    parameter_sets = read_parameters(n)
    num_simulations = parameter_sets.shape[0]
    part_sets = np.zeros((num_simulations, n))
    sign_sets = np.zeros((num_simulations, 2 * n ** 2))
    if m[1] == 'groupwise':
        part_sets = read_partitions(n)
    if m[3] != 0:
        sign_sets = read_sign(n)

    states = computeStates(n)
    mu = np.ones(n)
    marginal_operator = np.array([[1 if (t + 1) in s.types else 0 for s in states] for t in range(n)])
    steady_states = np.ones((num_simulations, 2 ** n))

    # Open a csv file to write the rates of transition between states at the simulated steady states
    filename = '_'.join(m) + '_{}types_0vts_VE0_trans_matrix.csv'.format(n)
    writer = csv.writer(open(filename, 'w', newline=''))

    # Simulate pre-vaccination steady states for each parameter set
    previous_run_times = []
    for j in range(num_simulations):
        start_time = time()

        # Get the type-specific transmissibilities
        beta = parameter_sets[j, :n]

        # Derive the required parameters
        k_matrix, h_matrix, states, partition = derive_param(parameter_sets[j], sign_sets[j], part_sets[j], n, 0, m)

        # Simulate the equilibrium corresponding to the j-th parameter set
        ss = get_steady_state(n, states, k_matrix, h_matrix, beta, mu, marginal_operator, True, m[1], partition)
        steady_states[j, :] = ss[0]
        row = np.reshape(ss[1], np.prod(ss[1].shape))
        writer.writerow(row)

        end_time = time()

        # Save the time used to simulate the first 100 parameter set and estimate the remaining siumlation time
        if j < 100:
            # save the time used to simulate the j-th parameter set
            previous_run_times.append(end_time - start_time)
        if j % int(num_simulations / 10) == 0:
            # print each time 10% of all parameter sets are simulated
            print(j, end_time - start_time)
        if j == 100:
            # print the estimated time until all parameter sets are simulated
            print([int(x) for x in strftime("%Y,%m,%d,%H,%M,%S").split(',')][3:5])
            print(sum(previous_run_times) / 100 * num_simulations / 3600, ' hours to go.')

    # write the simulated steady states in a csv file
    filename = '_'.join(m) + '_{}types_0vts_VE0_prevalence.csv'.format(n)
    df = pd.DataFrame(steady_states)
    df.to_csv(filename)


def simulate_post_vaccination(n, n_vt, m):
    # """
    # :param n: number of types
    # :param n_vt: number of vaccine types
    # :param m: interaction structure
    # """

    # param parameter_sets: np.array of parameter sets (transmissibilities and interaction parameters)
    # part_sets: np.array of partition (clusters/groups of types) sets
    # sign_sets: np.array of sign parameters
    # num_simulations: number parameter sets to simulate from
    # states: list of infection states
    # mu: the baseline per capita type-specific clearance rate
    # marginal operator: matrix for left-multiplying with the state variable to get the marginal prevalence
    # steady_states: np.array for saving the simulated steady state prevalence per infection state
    parameter_sets = read_parameters(n)
    num_simulations = parameter_sets.shape[0]
    part_sets = np.zeros((num_simulations, n))
    sign_sets = np.zeros((num_simulations, 2 * n ** 2))
    if m[1] == 'groupwise':
        part_sets = read_partitions(n)
    if m[3] != 0:
        sign_sets = read_sign(n)

    n_nvt = n - n_vt
    full_states = computeStates(n)
    reduced_states = computeStates(n_nvt)
    n_full_states = len(full_states)
    n_reduced_states = len(reduced_states)

    # matrix to translate a state in the reduced system to the full system
    state_translater = np.zeros((n_full_states, n_reduced_states))
    for i in range(n_reduced_states):
        reduced_state = reduced_states[i]
        state_id = getState(list(np.array(reduced_state.types) + n_vt), full_states).state_id
        state_translater[state_id, i] = 1

    mu = np.ones(n - n_vt)
    marginal_operator = np.array([[1 if (t + 1) in s.types else 0 for s in reduced_states] for t in range(n - n_vt)])
    steady_states = np.ones((num_simulations, 2 ** n))
    # Simulate pre-vaccination steady states for each parameter set
    previous_run_times = []
    for j in range(num_simulations):
        start_time = time()

        # Get the type-specific transmissibilities
        beta = parameter_sets[j, n_vt:n]

        # Derive the required parameters
        k_matrix, h_matrix, states, partition = derive_param(parameter_sets[j], sign_sets[j], part_sets[j], n, n_vt, m)

        # Simulate the equilibrium corresponding to the j-th parameter set
        ss = get_steady_state(n-n_vt, states, k_matrix, h_matrix, beta, mu, marginal_operator, False, m[1], partition)
        steady_states[j, :] = state_translater.dot(ss)

        end_time = time()

        # Save the time used to simulate the first 100 parameter sets and estimate the remaining simulation time
        if j < 100:
            # save the time used to simulate the j-th parameter set
            previous_run_times.append(end_time - start_time)
        if j % int(num_simulations / 10) == 0:
            # print each time 10% of all parameter sets are simulated
            print(j, end_time - start_time)
        if j == 100:
            # print the estimated time until all parameter sets are simulated
            print([int(x) for x in strftime("%Y,%m,%d,%H,%M,%S").split(',')][3:5])
            print(sum(previous_run_times) / 100 * num_simulations / 3600, ' hours to go.')

    # write the simulated steady states in a csv file
    filename = '_'.join(m) + '_{}types_{}vts_VE1_prevalence.csv'.format(n, n_vt)
    df = pd.DataFrame(steady_states)
    df.to_csv(filename)


if __name__ == '__main__':
    # generate required parameter sets
    Paper.parameters_generation.main()

    # choose interaction structure
    #       - mode: with/without interaction in clearance 'acq' or 'acq_cl'
    #       - multi: group- or type-wise multiplicative 'groupwise' or 'typewise'
    #       - recip: structured reciprocal: 'structured' or 'unstructured'
    #       - epsilon: ratio of reciprocal interactions (epsilon): [0.2, 0.5, 1]
    # model is the complete collection of choices
    max_num_types = 6
    mode = 'acq'            # 'acq' or 'acq_cl'
    multi = 'typewise'      # 'typewise' or 'groupwise'
    recip = 'structured'    # 'structured' or 'unstructured'
    epsilon = '5'           # '0', '2' or '5' for 0, 0.2 or 0.5, respectively
    model = [mode, multi, recip, epsilon]

    print('Simulating model={} ...'.format(model))

    for num_types in range(2, max_num_types + 1):
        # Simulate the pre-vaccination equilibria in the system with the number of pathogen types being n_types
        print('Simulating pre-vaccination equilibria #types={} ...'.format(num_types))
        simulate_pre_vaccination(num_types, model)

        # Simulate the post-vaccination steady states in the system with various number of non-vaccine types
        for num_vt in range(1, num_types):
            print('Simulating post-vaccination equilibria with #types={}, #VTs={} ...'.format(num_types, num_vt))
            simulate_post_vaccination(num_types, num_vt, model)

    print('Simulation completed.')
