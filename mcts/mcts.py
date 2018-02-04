import math
import os

import numpy as np

from mcts.node import Node


class MCTS:
    def __init__(self, state, max_steps, network_wrapper, s_size, store_mcts=False, scalar=1 / math.sqrt(2.0),
                 worker_name='default'):
        self.root = Node(state, network_wrapper, s_size)
        self.max_steps = max_steps
        self.network_wrapper = network_wrapper
        self.scalar = scalar
        self.store_mcts = store_mcts
        self.tree_path = 'trees/{}'.format(worker_name)
        self.leaves = dict()

        if self.store_mcts and not os.path.exists(self.tree_path):
            os.mkdir(self.tree_path)

    def search(self, budget, episode=0, step=0):
        count = 0

        while count < budget:
            frontier = self.select()
            count += len(frontier)
            # print('Frontier len: {}, leaves length: {}, count: {}'.format(len(frontier), len(leaves), count))

            for front in frontier:
                value = self.simulate(front)
                self.backpropagate(front, value)

        if self.store_mcts:
            try:
                self.draw('{}/{}_mcts_tree_{}.txt'.format(self.tree_path, episode, step))
            except RecursionError:
                print('Tree too big to draw using recursion')
                self.store_mcts = False

        # Select best child and update new root
        action_dist = self.root.get_action_dist()
        best_action = np.argmax(action_dist)

        new_root = list(filter(lambda child: child.action == best_action, self.root.children))[0]

        # print(action_dist)
        # print('Best action selected: {}'.format(best_action))
        # print('New root: {}'.format(new_root.id))
        # print('Leaves: {}'.format(len(self.leaves)))

        # The new root must have the new environment
        self.replace_root(new_root)

        return action_dist, best_action

    # The goal is to either expand the node, or to select the best child of the node. Only applicable to the root
    def select(self):
        # Select the most promising node of all
        # 1. clear up list of leaves that has already been expanded
        # 2. Select the best leaf to expand
        # 3. Expand the leaf

        if not self.root.fully_expanded():
            try:
                nodes = self.root.expand_all()
            except TypeError as e:
                print('Root children: {}, root id: {}, # root parents: {}, has parent: {}'.format(
                    len(self.root.children), self.root.id, len(self.root.parent_ids), self.root.parent is not None))
                print(e)
                exit(0)

            self.leaves.pop(self.root.id, None)
        else:
            best_leaf = self.best_child(self.leaves.values())

            if best_leaf is None:
                print('Leaves: {}, root children: {}, root done: {}'.format(len(self.leaves), len(self.root.children), self.root.done))

            # Pop the child from the dict
            self.leaves.pop(best_leaf.id, None)
            nodes = best_leaf.expand_all()

        for node in nodes:
            self.leaves[node.id] = node

        return nodes

    def best_child(self, children):
        # best_score = 1e-5
        best_child = sorted(children, key=lambda child: child.UBT(self.root.visits))

        if not any(best_child):
            return None

        best_child = best_child[-1]

        # for child in children:
        #     score = child.UBT(self.root.visits)
        #
        #     if score == best_score:
        #         best_children.append(child)
        #     if score > best_score:
        #         best_children = [child]
        #         best_score = score
        #
        # return random.choice(best_children)

        return best_child

    def simulate(self, front):
        """
        We have left out the ability to simulate ahead, mostly for simplicity
        :param front:
        :return: The value of the node
        """

        a, v = front.eval()

        return v

    #
    # def apply_previous_steps(self, front, game_env):
    #     actions = []
    #     while front.parent is not None:
    #         actions.append(front.action)
    #         front = front.parent
    #
    #     # reverse actions such that we apply them from the root
    #     actions.reverse()
    #
    #     state = self.root.state
    #     done = False
    #
    #     for a in actions:
    #         state, _, done, _ = game_env.step(a)
    #         state = process_frame(state, self.root.state.shape[0])
    #
    #     return state, done

    def replace_root(self, new_root):
        self._terminate_leaves(new_root.id)
        new_root.parent = None
        self.root = new_root

    def _terminate_leaves(self, new_root_id):
        leaves = list(self.leaves.values())
        for leaf in leaves:
            if new_root_id not in leaf.parent_ids and new_root_id is not leaf.id:
                self.leaves.pop(leaf.id, None)

            leaf.parent_ids.remove(self.root.id)

    def draw(self, file_name):
        content = ['digraph {', 'node [rx=5 ry=5 labelStyle="font: 300 14px Helvetica"]',
                   'edge [labelStyle="font: 300 14px Helvetica"]']

        content = content + self.root.draw(self.root.visits)
        content += ['}']
        file_content = '\n'.join(content)

        with open(file_name, 'w') as file:
            file.write(file_content)

    @staticmethod
    def backpropagate(node, value):
        while node is not None:
            node.update(value)
            node = node.parent
