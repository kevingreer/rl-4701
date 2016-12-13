import random

import numpy as np

LEFT, BRAKE, RIGHT = 0, 1, 2

#############################################
# DISCLAIMER: this class will not function because we modified the gym environment to create the `peek_step` function,
# which this class uses. We do not include that modification of gym here.
#############################################
class Agent(object):
    def __init__(self, model, action_space):
        self.model = model
        self.action_space = action_space

    def get_action(self, env):
        best_r = -1000
        best_a = 2

        for a in self.action_space:
            x, _, _, _ = env.peek_step(a)
            r = self.model.get_output(np.array([x]))
            if r > best_r:
                best_r = r
                best_a = a
            elif r == best_r:
                best_a = random.choice([a, best_a])
        return best_a

class SimpleAgent(object):
    def __init__(self):
        self.going_left = True
        self.prev_x = 0.

    def get_action(self, obs):
        x = obs[0]
        if self.going_left:
            if self.prev_x < x:
                self.going_left = False
                result = RIGHT
            else:
                result = LEFT
        else:
            if self.prev_x > x:
                self.going_left = True
                result = LEFT
            else:
                result = RIGHT
        self.prev_x = x
        return result





