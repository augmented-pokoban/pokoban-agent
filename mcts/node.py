from random import shuffle


class Node:
    def __init__(self, state, action_count, parent=None, action=None):
        self.visits = 1
        self.value = 0.0
        self.state = state
        self.children = []
        self.parent = parent
        self.action = action
        self.child_actions = set()
        self.unexplored_actions = list(range(action_count))
        shuffle(self.unexplored_actions)

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child_node):
        self.children.append(child_node)
        self.child_actions.add(child_node.action)

    def update(self, value):
        self.value += value
        self.visits += 1

    def fully_expanded(self):
        return not any(self.unexplored_actions)

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f; action: %d" % (len(self.children), self.visits, self.value,
                                                                        self.action)
        return s
