class MovesWrapper:
    def __init__(self, game):
        """

        :type game: ExpertMoves
        """
        self.game = game
        self.next_index = 0

    def has_next(self):
        # if there is at least one transition and we have not taken one yet, we are good
        # if we have taken the first, then we are on to just take the current index,

        if not any(self.game.transitions):
            return False
        elif len(self.game.transitions) == self.next_index:
            return False
        else:
            return True

    def get_next(self):
        """
        Assumes that has_next() validates True
        :return: State, Transition
        """

        if self.next_index == 0:
            state, trans = self.game.initial, self.game.transitions[self.next_index]
        else:
            state, trans = self.game.transitions[self.next_index - 1].state, self.game.transitions[self.next_index]

        self.next_index += 1
        return state, trans
