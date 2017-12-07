from env import api
from env.expert_moves import ExpertMoves


class ApiLoader:
    def __init__(self, api_method, name):
        self._api_method = api_method
        self._name = name
        self._total = 0
        self._next_batch = 0
        self._batch_size = 1000
        self._loaded = None

    def get_next(self):
        """
        Expects that has_next has validated to true beforehand
        :return: ExpertMoves
        """
        if self._loaded is None or not any(self._loaded):
            # First retrieval from API
            self._load_next_batch()

        game = self._loaded.pop()
        return ExpertMoves(api.get_expert_game(game['fileRef']))

    def has_next(self):
        """
        Validates that there exist more data.
        :return: True if get_next() can return another game.
        """
        if self._loaded is None:
            # Assume that there is data on the backend
            return True
        elif any(self._loaded):
            return True
        elif self._next_batch < self._total:
            return True
        else:
            return False

    def _load_next_batch(self):
        """
        Retrieves next page, stores it in object and updates the next_batch.
        """

        resp = self._api_method(self._next_batch, self._batch_size)
        self._loaded = resp['data']
        self._total = resp['total']
        print('{}: Next batch with skip: {}, take: {}, total: {}'.format(self._name, self._next_batch, self._batch_size,
                                                                         self._total))
        self._next_batch += self._batch_size

