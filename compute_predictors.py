import numpy as np
from Paper.infection_state import getState


def compute_odds_based_predictors(steady_state, states, vt, nvt_):
    """
    computing type-specific hazard-based predictors
    :param steady_state: a list of prevalence of infection states
    :param states: a list of state objects
    :param vt: a list of vaccine types
    :param nvt_: a integer giving the non-vaccine type at interest
    :return: type-specific hazard-based- predictor for type replacement by the non-vaccine type at interest
    """
    odds1_numerator = 0
    odds1_denominator = 0
    odds2_numerator = 0
    odds2_denominator = 0

    for state in states:
        if len(set(state.types) & set(vt)) > 0:
            if len(set(state.types) & set([nvt_])) > 0:
                odds1_numerator += steady_state[state.state_id]
            else:
                odds1_denominator += steady_state[state.state_id]
        else:
            if len(set(state.types) & set([nvt_])) > 0:
                odds2_numerator += steady_state[state.state_id]
            else:
                odds2_denominator += steady_state[state.state_id]

    predictor = (odds1_numerator / odds1_denominator) / (odds2_numerator / odds2_denominator)

    return predictor


def compute_hazard_based_predictor(steady_state, trans_matrix, states, vt, nvt_):
    """
    computing type-specific hazard-based predictors
    :param steady_state: a list of prevalence of infection states
    :param trans_matrix: transition matrix (hazards)
    :param states: a list of state objects
    :param vt: a list of vaccine types
    :param nvt_: a integer giving the non-vaccine type at interest
    :return: type-specific hazard-based- predictor for type replacement by the non-vaccine type at interest
    """
    entries = np.zeros((2, 4))

    for state in states:
        if len(set(state.types) & set(vt)) > 0 and len(set(state.types) & set([nvt_])) == 0:
            state_nvt = getState(state.types + [nvt_], states)
            # hazards of acquiring the non-vaccine type at interest from states:
            #   - containing at least a vaccine type
            #   - not containing the non-vaccine type at interest
            entries[0, 0] += steady_state[state.state_id] * trans_matrix[state.state_id, state_nvt.state_id]
            entries[1, 0] += steady_state[state.state_id]
            # hazards of clearing the non-vaccine type at interest from states:
            #   - containing at least a vaccine type
            #   - containing the non-vaccine type at interest
            entries[0, 1] += steady_state[state_nvt.state_id] * trans_matrix[state_nvt.state_id, state.state_id]
            entries[1, 1] += steady_state[state_nvt.state_id]
        elif len(set(state.types) & set(vt)) == 0 and len(set(state.types) & set([nvt_])) == 0:
            state_nvt = getState(state.types + [nvt_], states)
            # hazards of acquiring the non-vaccine type at interest from states:
            #   - not containing any vaccine type
            #   - not containing the non-vaccine type at interest
            entries[0, 2] += steady_state[state.state_id] * trans_matrix[state.state_id, state_nvt.state_id]
            entries[1, 2] += steady_state[state.state_id]
            # hazards of clearing the non-vaccine type at interest from states:
            #   - not containing any vaccine type
            #   - containing the non-vaccine type at interest
            entries[0, 3] += steady_state[state_nvt.state_id] * trans_matrix[state_nvt.state_id, state.state_id]
            entries[1, 3] += steady_state[state_nvt.state_id]

    # compute all weighted averages
    combined_entries = entries[0]/entries[1]
    predictor = (combined_entries[0]/combined_entries[2]) / (combined_entries[1]/combined_entries[3])
    return predictor

