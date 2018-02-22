class ExpertMoves:
    def __init__(self, mapping):
        self.initial = State(mapping['initial'])
        self.transitions = [Transition(x) for x in mapping['transitions']]
        self.level = mapping['level']


    def get_transition(self, index):
        if len(self.transitions) == index:
            trans = self.transitions[index-1]
            trans.done = True
            return trans

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
        self.done = transition['done']
        self.success = transition['success']
        self.reward = 10.0 if self.done else transition['reward']
        self.action = transition['action']
        self.state = State(transition['state'])



class PokobanObject:
    def __init__(self, obj):
        self.col = obj['col']
        self.row = obj['row']
        self.letter = obj['letter']
