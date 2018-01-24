import math
import os
from mcts.node import Node


class MCTS:

    def __init__(self, state, max_steps, game_env, network_wrapper, s_size, store_mcts=False, scalar=1 / math.sqrt(2.0)):
        self.root = Node(state, game_env, network_wrapper, s_size)
        self.max_steps = max_steps
        self.game_env = game_env
        self.network_wrapper = network_wrapper
        self.scalar = scalar
        self.store_mcts = store_mcts
        self.tree_path = 'trees/'
        if self.store_mcts and not os.path.exists(self.tree_path):
            os.mkdir(self.tree_path)

    def search(self, budget, episode=0, step=0):
        leaves = self.root.get_leaves_in_tree()
        count = 0

        while count < budget:
            frontier, leaves = self.select(leaves)
            count += len(frontier)

            for front in frontier:
                value = self.simulate(front)
                self.backpropagate(front, value)

        if self.store_mcts:
            self.draw(self.tree_path + '{}_mcts_tree_{}.txt'.format(episode, step))

        new_root = self.best_child(self.root.children)
        new_root.parent = None

        self.root.terminate(ignore_action=new_root.action)
        self.root = new_root

        return self.root.get_action_dist()

    # The goal is to either expand the node, or to select the best child of the node. Only applicable to the root
    def select(self, leaves):
        # Select the most promising node of all
        # 1. clear up list of leaves that has already been expanded
        # 2. Select the best leaf to expand
        # 3. Expand the leaf

        if not self.root.fully_expanded():
            nodes = self.root.expand_all()
        else:
            leaves = list(filter(lambda leaf: not leaf.fully_expanded(), leaves))
            best_leaf = self.best_child(leaves)
            nodes = best_leaf.expand_all()

        leaves += nodes
        return nodes, leaves

    def best_child(self, children):
        # best_score = 1e-5
        best_child = sorted(children, key=lambda child: child.UBT(self.root.visits))
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

    def terminate(self):
        self.root.terminate()

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



