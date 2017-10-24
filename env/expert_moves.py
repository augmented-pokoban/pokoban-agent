class ExpertMoves:

    def __init__(self, mapping):
        self.initial = State(mapping['initial'])
        self.transitions = [Transition(x) for x in mapping['transitions']]
        self.level = mapping['level']
        self.value = sum(map(lambda trans: trans.reward, self.transitions))

    def get_transition(self, index):
        return self.transitions[index]


class State:

    def __init__(self, state):
        self.agents = [PokobanObject(x) for x in state['agents']]
        self.walls = [PokobanObject(x) for x in state['walls']]
        self.boxes = [PokobanObject(x) for x in state['boxes']]
        self.goals = [PokobanObject(x) for x in state['goals']]
        self.dims = state['dimensions']


class Transition:

    def __init__(self, transition):
        self.reward = transition['reward']
        self.success = transition['success']
        self.action = transition['action']
        self.state = State(transition['state'])
        self.done = transition['done']


class PokobanObject:

    def __init__(self, object):
        self.col = object['col']
        self.row = object['row']
        self.letter = object['letter']