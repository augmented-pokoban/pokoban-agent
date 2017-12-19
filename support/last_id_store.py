import os


class IdStore:
    def __init__(self, name, folder='last_ids'):
        self.name = name
        self.folder = folder
        self.path = os.path.join(self.folder, self.name + '.id')

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def write_id(self, level_id):
        with open(self.path, 'w') as file:
            file.write(level_id)

    def get_id(self):
        if not os.path.exists(self.path):
            return None
        with open(self.path, 'r') as file:
            return file.read()

