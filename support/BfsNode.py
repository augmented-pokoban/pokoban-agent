class BfsNode:
    def __init__(self, state, depth, rnn_state=None, value=0):
        self.value = value
        self.state = state
        self.depth = depth
        self.rnn_state = rnn_state

    def get_hash(self):
        return hash(self.state.tostring())

    def __lt__(self, other):
        return self.value < other.value

