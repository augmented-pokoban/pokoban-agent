class BfsNode:
    def __init__(self, state, depth):
        self.state = state
        self.depth = depth

    def get_hash(self):
        return hash(self.state.tostring())
