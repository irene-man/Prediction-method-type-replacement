import itertools


class State:
    def __init__(self, state_id, types, clusters=[]):
        self._state_id = state_id
        self._types = types
        self._clusters = clusters

    @property
    def state_id(self):
        return self._state_id

    @property
    def types(self):
        return self._types

    @property
    def clusters(self):
        return self._clusters


def computeStates(n_types, partition=None):
    # n_types: number of types
    # partition: list of lists of types belonging to the same cluster (group) of types

    type_names = set(range(1, n_types + 1))
    states = []
    state_id = 0
    m_types = n_types # maximum number of types in a state
    for i in range(m_types + 1):
        for types in itertools.combinations(type_names, i):
            if partition == None:
                states.append(State(state_id, list(types)))
            else:
                clusters = []
                for i in range(len(partition)):
                    part = partition[i]
                    if len(set(part) & set(types)) > 0:
                        clusters.append(i + 1)
                states.append(State(state_id, list(types), clusters))
            state_id += 1
    return states

def getState(types, states):
    for state in states:
        if len(types) != len(state.types):
            continue
        if set(state.types) == set(types):
            return state


def setClusters(state, partition):
    clusters = []
    for i in range(len(partition)):
        part = partition[i]
        if len(set(part) & set(state._types)) > 0:
            clusters.append(i + 1)
    state._clusters = clusters