import json
from pprint import pprint
import os
import random
from env.expert_moves import ExpertMoves


class Data:

    def __init__(self):
        self.path = 'env/expert-moves/'
        self.file_names = os.listdir(self.path)

    def load_random(self):
        file_name = random.choice(self.file_names)
        return ExpertMoves(self._load_file(file_name))

    def print_random(self):
        rand = self.load_random()
        pprint(rand)

    def _load_file(self, file_name):
        with open(self.path + file_name) as data_file:
            data = json.load(data_file)
        return data
