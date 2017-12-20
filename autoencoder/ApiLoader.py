from env import api
from env.expert_moves import ExpertMoves
from support.last_id_store import IdStore


class ApiLoader:
    def __init__(self, api_method, name):
        self._api_method = api_method
        self._name = name
        self._batch_size = 1000
        self._has_more = True
        self._loaded = None
        self._id_store = IdStore(name)
        self._last_id = self._id_store.get_id()

    def get_next(self):
        """
        Expects that has_next has validated to true beforehand
        :return: ExpertMoves
        """
        if self._loaded is None or not any(self._loaded):
            # First retrieval from API
            self._load_next_batch()

        if not any(self._loaded):
            return None

        game = self._loaded.pop()
        return ExpertMoves(api.get_expert_game(game['fileRef']))

    def has_next(self):
        """
        Validates that there exist more data.
        :return: True if get_next() can return another game.
        """
        if self._loaded is None or self._last_id is None:
            # Assume that there is data on the backend
            return True
        elif any(self._loaded):
            return True
        elif self._has_more:
            return True
        else:
            return False

    def _load_next_batch(self):
        """
        Retrieves next page, stores it in object and updates the next_batch.
        """
        resp = self._api_method(self._last_id, self._batch_size, order='asc')
        self._loaded = resp['data']
        self._total = resp['total']
        self._has_more = len(self._loaded) == self._batch_size

        if any(self._loaded):
            self._last_id = self._loaded[-1]['_id']
            self._id_store.write_id(self._last_id)  # Write last id to store

        print('{}: Next batch with skip: {}, id: {}, total: {}'.format(self._name, self._batch_size, self._last_id,
                                                                       self._total))

