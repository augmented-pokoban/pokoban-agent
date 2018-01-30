from random import shuffle
from helper import process_frame
import math
import uuid


class Node:
    def __init__(self, state, game_env, network_wrapper, s_size, parent=None, action=None, action_prop=1):
        self.id = 'n_' + str(uuid.uuid4().hex)
        self.game_env = game_env  # Notice that this gets terminated for all except leaves and root
        self.network_wrapper = network_wrapper
        self.s_size = s_size
        self.visits = 0
        self.value = 0.0
        self.action_prop = action_prop
        self.state = state
        self.children = []
        self.parent = parent
        self.action = action
        self.child_actions = set()
        self.unexplored_actions = list(range(game_env.get_action_count()))
        shuffle(self.unexplored_actions)
        self.p_a, _ = self.eval()
        self.depth = 0
        self.parent_ids = set()

        if parent is not None:
            parent.add_child(self)
            self.depth = parent.depth + 1
            # Append parent id
            self.parent_ids = set(parent.parent_ids)
            self.parent_ids.add(parent.id)

    def add_child(self, child_node):
        self.children.append(child_node)
        self.child_actions.add(child_node.action)

    def update(self, value):
        self.value += value
        self.visits += 1

    def expand_all(self):
        """
        Assume that the fully expanded call returns false
        :return:
        """
        # Copy environment here
        new_env = self.game_env.copy()
        while any(self.unexplored_actions):

            action = self.unexplored_actions.pop()
            state, r, done, success = new_env.step(action)
            if success:
                if not self.network_wrapper.has_running():
                    self.network_wrapper.eval(self.state)

                new_wrapper = self.network_wrapper.get_next_wrapper()
                Node(process_frame(state, self.s_size), new_env, new_wrapper, self.s_size, self, action,
                     self.p_a[action])

                new_env = self.game_env.copy()

        new_env.terminate()

        # Terminate because we don't need it anymore - and reset parent_ids, they have been used
        if not self._is_root():
            self.game_env.terminate()
        self.parent_ids = None

        return self.children

    def _is_root(self):
        return self.parent is None

    def terminate_env(self):
        self.game_env.terminate()

    def UBT(self, root_visits, scalar=1 / math.sqrt(2.0)):

        exploit = self.value / self.visits
        explore = self.action_prop * math.sqrt(2.0 * math.log(root_visits) / float(self.visits))
        score = exploit + scalar * explore
        return score

    def eval(self):
        a, v = self.network_wrapper.eval(self.state)
        return a, v

    def fully_expanded(self):
        return not any(self.unexplored_actions)

    def get_action_dist(self):
        prob = [0.0] * self.game_env.get_action_count()
        for child in self.children:
            # print('Root visits: {}, child visits: {}, child action: {}, children: {}'.format(self.visits, child.visits, child.action, len(self.children)))
            prob[child.action] = float(child.visits) / float(self.visits)

        return prob

    def draw(self, root_visits, prev_id=None):

        data = 'UBT: {}, Vis: {}, Val: {}, FE: {}'.format(self.UBT(root_visits), self.visits, self.value,
                                                          self.fully_expanded())
        content = [self.id + ' [label="{}"];'.format(data)]
        if prev_id is not None:
            content.append(
                '{} -> {} [label="{}"];'.format(prev_id, id, self.game_env.get_action_meanings()[self.action]))

        for child in self.children:
            content += child.draw(root_visits, id)

        return content

    def __repr__(self):
        s = "Children: %d; visits: %d; Value: %f; Action: %d" % (len(self.children), self.visits, self.value,
                                                                 self.action)
        return s
