from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
import random
import numpy as np
import gym
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

KPCA_DIMS = 100
HIDDEN1_SIZE = 512
HIDDEN2_SIZE = 256
LEARNING_RATE = 0.0001
STARTING_EPS = 0.5
GAMMA = 1.0

MINIMUM_EPS = 0.1
DAMP_FACTOR_EPS = 0.9
EPOCHS = 3000
MEM_LIMIT = 100000
BATCH_SIZE = 40
FRAME_LIMIT = 10000

WIN_STREAK = 20
WIN_THRESHOLD = 300
FAILS_ALLOWED = 5

RENDER = False

class QModel(object):
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        model = Sequential()
        model.add(Dense(HIDDEN1_SIZE, init='glorot_normal', input_shape=(KPCA_DIMS,)))
        model.add(LeakyReLU())

        model.add(Dense(HIDDEN2_SIZE, init='glorot_normal'))
        model.add(LeakyReLU())

        model.add(Dense(3, init='glorot_normal'))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        rms = RMSprop(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=rms)
        self.model = model

        observation_examples = np.array([self.env.observation_space.sample() for x in range(100000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.feature_map = RBFSampler(n_components=100, gamma=1., random_state=1)
        self.feature_map.fit(self.scaler.transform(observation_examples))

    def process(self, states):
        s_float32 = np.array(states).astype(np.float32)
        if len(s_float32.shape) == 1:
            s_float32 = np.expand_dims(s_float32, axis=0)
        s_float32 = self._scale_state(s_float32)
        s_float32 = self.feature_map.transform(s_float32)
        s_float32 = s_float32.astype(np.float32)
        return s_float32

    def _scale_state(self, s_float32):
        return self.scaler.transform(s_float32)

    def predict(self, s):
        phi = self.process(s)
        return self.model.predict(phi, batch_size=phi.shape[0])

    def train(self, x, y):
        phi = self.process(x)
        return self.model.fit(phi, y, batch_size=phi.shape[0], nb_epoch=1, verbose=0)

    def learn(self):
        failures = 0
        streak = 0
        started_streak = -1
        env = self.env
        epsilon = STARTING_EPS
        memory = []
        for epoch in range(EPOCHS):
            state = env.reset()
            done = False
            frame_count = 0
            while not done:
                if RENDER:
                    env.render()
                if random.random() < epsilon:
                    action = np.random.randint(0, 3)
                else:
                    qval = self.predict(state)
                    action = (np.argmax(qval))
                new_state, reward, done, _ = env.step(action)

                memory.append((state, action, reward, new_state))
                if len(memory) > MEM_LIMIT:
                    memory.pop(0)
                if len(memory) > 2*BATCH_SIZE:
                    minibatch = random.sample(memory, BATCH_SIZE)
                    X = map(lambda t: t[0], minibatch)
                    X_next = map(lambda t: t[3], minibatch)
                    oldq_vals = self.predict(np.array(X))
                    q_vals = self.predict(np.array(X_next))
                    maxes= np.amax(q_vals, axis=1)
                    func = lambda m, r: np.add(r, np.multiply(GAMMA, m))
                    vfunc = np.vectorize(func)
                    updates = vfunc(maxes, reward)
                    for i in range(BATCH_SIZE):
                        oldq_vals[i, minibatch[i][1]] = updates[i]
                    self.train(np.array(X), oldq_vals)
                state = new_state
                frame_count += 1

                if frame_count > FRAME_LIMIT:
                    failures += 1
                    if failures == FAILS_ALLOWED:
                        print "Experiment completed: failure in {} epochs".format(epoch)
                        exit()

            if epsilon > MINIMUM_EPS:
                epsilon *= DAMP_FACTOR_EPS

            if frame_count < WIN_THRESHOLD:
                if started_streak < 0:
                    started_streak = epoch - 1
                streak += 1
                print "Epoch {} finished after {} frames\tWin streak of {}".format(epoch, frame_count, streak)
                if streak == WIN_STREAK:
                    print "Experiment completed: took {} epochs".format(started_streak)
                    exit()
            else:
                print "Epoch {} finished after {} frames".format(epoch, frame_count)
                streak = 0
                started_streak = -1

QModel().learn()
