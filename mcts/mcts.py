import math
import random

from helper import process_frame
from mcts.node import Node


class MCTS:

    def __init__(self, state, max_steps, game_env, network_wrapper, scalar=1/math.sqrt(2.0)):
        self.root = Node(state, game_env.get_action_count())
        self.max_steps = max_steps
        self.game_env = game_env
        self.network_wrapper = network_wrapper
        self.scalar = scalar

    def search(self, budget):
        for budget in range(budget):
            front = self.select()
            value = self.simulate(front)
            self.backpropagate(front, value)

        # Update root by getting the best child.. Might need another method that selects the
        # most visited node
        # TODO: Do the above
        self.root = self.best_child()
        self.root.parent = None
        return self.root.action

    # The goal is to either expand the node, or to select the best child of the node. Only applicable to the root
    def select(self):
        if not self.root.fully_expanded():
            return self.expand_root()
        else:
            return self.best_child()

    def expand_root(self):
        next_action = self.root.unexplored_actions.pop()
        return Node(None, self.game_env.get_action_count(), self.root, next_action)

    # TODO: Investigate if this is the best method
    def best_child(self):
        best_score = 0.0
        best_children = []
        for child in self.root.children:
            exploit = child.value / child.visits
            explore = math.sqrt(2.0 * math.log(self.root.visits) / float(child.visits))
            score = exploit + self.scalar * explore
            if score == best_score:
                best_children.append(child)
            if score > best_score:
                best_children = [child]
                best_score = score

        return random.choice(best_children)

    def simulate(self, front, store=False):
        t = 0
        v = None
        game_env = self.game_env.copy(store=store)
        self.network_wrapper.start()

        # State here is a vector
        state, done = self.apply_previous_steps(front, game_env)
        front.state = state

        while not done and t < self.max_steps:
            a, v = self.network_wrapper.eval(state)
            state, _, done, _ = game_env.step(a)
            state = process_frame(state, self.root.state.shape[0])
            t += 1

        # Kill game on server
        game_env.terminate(description='Value of game: ' + str(v))
        return v

    def apply_previous_steps(self, front, game_env):
        actions = []
        while front.parent is not None:
            actions.append(front.action)
            front = front.parent

        # reverse actions such that we apply them from the root
        actions.reverse()

        state = self.root.state
        done = False

        for a in actions:
            state, _, done, _ = game_env.step(a)
            state = process_frame(state, self.root.state.shape[0])

        return state, done

    @staticmethod
    def backpropagate(node, value):
        while node is not None:
            node.update(value)
            node = node.parent



