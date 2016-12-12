import random

class RandomAgent(object):

    def __init__(self):
        self.name = 'Random'

    def get_action(self, moves, game, ideal=None):
        return random.choice(list(moves)) if moves else None