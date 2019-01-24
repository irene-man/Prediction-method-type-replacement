import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Paper.infection_state import getState


def get_steady_state(n_types, states, k_matrix, h_matrix, beta, mu, marginal_operator, pre_vacc, multi, partition):
    """
    Compute the prevalence of each state at the equilibrium
    :param n_types: number of types
    :param states: a list of infection states
    :param k_matrix: n_types**2 np.array of the interaction parameters in acquisition
    :param h_matrix: n_types**2 np.array of the interaction parameters in clearance
    :param beta: the type-specific transmission probability
    :param mu: the baseline type-specific clearance rate
    :param marginal_operator: matrix for left-multiplying with the state variable to get the marginal prevalence
    :param pre_vacc: boolean whether it is a pre- or post-vaccination simulation
    :param multi: 'typewise' or 'groupwise' multiplicative interaction structure
    :param partition: partition of groups(clusters) of types
    :return: steady_state: np.array of the prevalence of states at the end of simulation (equilibrium)
              trans_matrix: np.array of the transition rate at the end of simulation(equilibrium)
    """

    # compute the time invariant part of the transition matrices
    if multi != 'groupwise':
        acq_matrix = compute_acquisition_rates_time_invariant_part(n_types, states, k_matrix)
    else:
        acq_matrix = compute_acquisition_rates_time_invariant_part_group(n_types, states, k_matrix, partition)
    if multi != 'groupwise':
        cl_matrix = compute_clearance_rates(n_types, states, h_matrix, mu)
    else:
        cl_matrix = compute_clearance_rates_group(n_types, states, h_matrix, mu, partition)

    parameters = (n_types, states, beta, acq_matrix, cl_matrix, marginal_operator)
    n_states = len(states)
    state_0 = [1 / n_states] * n_states
    simulation_time = 1000
    t = np.linspace(0, simulation_time, 50)

    # simulate by solving the systems of ordinary differential equations numerically
    solution = odeint(solve_ode, state_0, t, args=parameters)
    steady_state = solution[-1, :]

    # plot trajectory of the marginal prevalence per type in time
    plot = False
    if plot:
        plot_trajectory_marginal(t, solution, n_types, marginal_operator)

    # return the simulated equilibrium prevalence (and transition rates)
    if pre_vacc:
        trans_matrix = compute_transition_matrix(steady_state, n_types, beta, acq_matrix, cl_matrix, marginal_operator)
        return steady_state, trans_matrix
    else:
        return steady_state


def compute_transition_matrix(state, n_types, beta, acq_matrix, cl_matrix, marginal_operator):
    """
    Compute the comprehensive transition matrix including all transition rates
    :param state: state variable
    :param n_types: number of types
    :param beta: the type-specific transmission probability
    :param acq_matrix: a list of acquisition matrices per type excluding the time varying part, the foi
    :param cl_matrix: a matrix for the clearance rate
    :param marginal_operator: matrix for left-multiplying with the state variable to get the marginal prevalence
    :return: dD: np.array of length len(states) of the derivatives of each state
    """
    foi = marginal_operator.dot(state) * beta
    transition_matrix = np.copy(cl_matrix)
    for i in range(n_types):
        transition_matrix += foi[i] * acq_matrix[i]
    return transition_matrix


def compute_acquisition_rates_time_invariant_part(n_types, states, k_matrix):
    """
    Compute the time invariant part of the acquisition rate
    :param n_types: number of types
    :param states: a list of infection states
    :param k_matrix: np.array of the pairwise interaction parameters in acquisition
    :return: output: list of length n_types. Each item in the list is a n_state**2 np.array.
       [j,k]-element of i-th matrix: acquisition interaction parameter from state j to state k.
    """
    n_states = len(states)
    output = []
    for t in range(1, n_types + 1):
        output_i = np.zeros((n_states, n_states))
        for end_state in states:
            if t not in end_state.types:
                continue
            start_state = getState([x for x in end_state.types if x != t], states)
            d = 1
            if len(start_state.types) > 0:
                d = np.prod([k_matrix[start_t - 1, t - 1] for start_t in start_state.types])
            output_i[start_state.state_id, end_state.state_id] = d
        output.append(output_i)
    return output


def compute_acquisition_rates_time_invariant_part_group(n_types, states, k_matrix, partition):
    """
    Compute the time invariant part of the acquisition rate of the all or nothing model
    :param n_types: number of types
    :param states: a list of state objects
    :param k_matrix: np.array of the pairwise interaction parameters in acquisition between clusters
    :param partition: list of lists of types belonging to the same cluster
    :return: output: list of length n_types. Each item in the list is a n_state**2 np.array.
       [j,k]-element of i-th matrix: acquisition interaction parameter from state j to state k.
    """
    n_states = len(states)
    output = []
    for t in range(1, n_types + 1):
        type_cluster = np.where([t in part for part in partition])[0][0] + 1

        output_i = np.zeros((n_states, n_states))
        for end_state in states:
            if t not in end_state.types:
                continue
            # if type is in endState
            state_state = getState([x for x in end_state.types if x != t], states)
            d = 1
            if len(state_state.types) > 0:
                d = np.prod([k_matrix[startCluster - 1, type_cluster - 1] for startCluster in state_state.clusters])
            output_i[state_state.state_id, end_state.state_id] = d
        output.append(output_i)
    return output


def compute_clearance_rates(n_types, states, h_matrix, mu):
    """
    Compute the clearance rate
    :param n_types: number of types
    :param states: a list of state objects
    :param h_matrix: np.array of the pairwise interaction parameters in clearance
    :param mu: the baseline type-specific clearance rate
    :return: output: a n_state**2 np.array.
        The [j,k]-element the clearance rate from state j to state k including the interaction parameters.
    """
    num_states = len(states)
    output = np.zeros((num_states, num_states))
    for t in range(1, n_types + 1):
        for start_state in states:
            if t not in start_state.types:
                continue
            end_state = getState([x for x in start_state.types if x != t], states)
            d = mu[t - 1]
            if len(end_state.types) > 0:
                d = np.prod([h_matrix[end_t - 1, t - 1] for end_t in end_state.types]) * mu[t - 1]
            output[start_state.state_id, end_state.state_id] = d
    return output


def compute_clearance_rates_group(n_types, states, h_matrix, mu, partition):
    """
    Compute the clearance rate
    :param n_types: number of types
    :param states: a list of state objects
    :param h_matrix: np.array of the pairwise interaction parameters in clearance
    :param mu: the baseline type-specific clearance rate
    :param partition: partition of groups (clusters) of types
    :return: output: a n_state**2 np.array.
    The [j,k]-element the clearance rate from state j to state k including the interaction parameters.
    """
    num_states = len(states)
    output = np.zeros((num_states, num_states))
    for t in range(1, n_types + 1):
        type_cluster = np.where([t in part for part in partition])[0][0] + 1
        for startState in states:
            if t not in startState.types:
                continue
            end_state = getState([x for x in startState.types if x != t], states)
            d = mu[t - 1]
            if len(end_state.types) > 0:
                d *= np.prod([h_matrix[end_clus - 1, type_cluster - 1] for end_clus in end_state.clusters])
            output[startState.state_id, end_state.state_id] = d
    return output


def solve_ode(state, t, n_types, states, beta, acq_matrix, cl_matrix, marginal_operator):
    """
    Loop for solving the Ode system in which the derivative is given by dD
    :param state: state variable
    :param t:
    :param n_types: number of types
    :param states: a list of state objects
    :param beta: the type-specific transmission probability
    :param acq_matrix: a list of acquisition matrices per type excl. the time varying part, the foi
    :param cl_matrix: a matrix for the clearance rate
    :param marginal_operator: matrix for left-multiplying with the state variable to get the marginal prevalence
    :return: dD: np.array of length len(states) of the derivatives of each state
    """
    q = compute_transition_matrix(state, n_types, beta, acq_matrix, cl_matrix, marginal_operator)
    n_states = len(states)
    transition_matrix = np.dot(np.transpose(q), state) - np.dot(q, np.ones(n_states)) * state
    return transition_matrix


def plot_trajectory_marginal(t, solution, n_types, marginal_operator):
    """
    Plot the simulated trajectories of the marginal prevalence per type
    :param t: the time points at which solution are solved at
    :param solution: the prevalence of each state at time points in t
    :param n_types: number of types
    :param marginal_operator: matrix for left-multiplying with the state variable to get the marginal prevalence
    :return: a plot of the marginal of the prevalence per type
    """
    marginal = np.dot(marginal_operator, np.transpose(solution))
    plt.figure()
    title = 'Marginals'
    plt.title(title, fontsize=15)
    plt.xlabel('t')
    colors = ['red',
              'blue',
              'green',
              'purple',
              'brown',
              'cyan',
              'black',
              'cyan',
              'magenta',
              'yellow']
    colors = colors[:n_types]
    labels = ["Type {%d}" % t_ for t_ in range(1, n_types + 1)]
    for i in range(len(colors)):
        plt.plot(t, marginal[i], '-', color=colors[i], label=labels[i])
    plt.ylim((0, 1))
    plt.legend()
    plt.title('Marginal prevalence per type')
    plt.show()
