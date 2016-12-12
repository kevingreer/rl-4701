import tensorflow as tf
import gym
from agent import Agent, SimpleAgent
import numpy as np
import os
import sys

CHECKPOINT_DIR = 'checkpoints/'
FRAME_LIMIT = 200
TRAINING_EPISODES = 1000
BOOTSTRAP_EPISODES = 1000
CHECKPOINT_INTERVAL = 50

def connected_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='biases')
        return activation(tf.matmul(x, W) + b)

class TdModel(object):
    def __init__(self, sess, restore=False):
        self.sess = sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.lamda = tf.constant(0.7)
        self.alpha = tf.constant(1.0)

        self.x = tf.placeholder('float', [1, 2], name='x')
        self.output_next = tf.placeholder('float', [1, 1], name='output')

        hidden = connected_layer(self.x, [2, 50], tf.nn.relu, name='hidden')
        self.output = connected_layer(hidden, [50, 1], tf.sigmoid, name='output')

        trainable_vars = tf.trainable_variables()
        gradients = tf.gradients(self.output, trainable_vars)

        delta_op = tf.reduce_sum(self.output_next - self.output, name='delta')
        update_ops = [self.global_step.assign_add(1)]
        with tf.variable_scope('update_ops'):
            # There is a trace (and update) for every variable
            for gradient, variable in zip(gradients, trainable_vars):
                with tf.variable_scope('/e_trace'):
                    e_trace = tf.Variable(tf.zeros(gradient.get_shape()), trainable=False, name="e_trace")
                    e_trace_op = e_trace.assign((self.lamda * e_trace) + gradient)  # from sutton appendix

                update_op = variable.assign_add(self.alpha * delta_op * e_trace_op)
                update_ops.append(update_op)

        self.train_op = tf.group(*update_ops, name='train')
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)
        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        return self.sess.run(self.output, feed_dict={self.x: x})

    def reward_to_output(self, r):
        return np.add(1, np.divide(r, FRAME_LIMIT))

    def train(self):
        env = gym.make('MountainCar-v0')
        agent = Agent(self, [0, 2])
        for episode in range(TRAINING_EPISODES):
            print "Episode {}".format(episode)
            total_reward = np.float32(0.0)
            ob = np.array([env.reset()])
            done = False
            for t in range(FRAME_LIMIT):
                env.render()
                action = agent.get_action(env)
                ob_next, _, done, _ = env.step(action)
                output_next = self.get_output(np.array([ob_next]))

                self.sess.run(self.train_op, feed_dict={self.x: ob, self.output_next: output_next})

                total_reward = np.subtract(total_reward, np.float32(1.0))
                ob = np.array([ob_next])
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
            if not done:
                total_reward = np.multiply(total_reward, 2)
            self.sess.run(self.train_op, feed_dict={self.x: ob,
                                                    self.output_next: np.array([[self.reward_to_output(total_reward)]])})

            if ((episode + 1) % CHECKPOINT_INTERVAL == 0) or (episode + 1) == TRAINING_EPISODES:
                checkpoint_file = os.path.join(CHECKPOINT_DIR, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, global_step=self.global_step)

    def bootstrap(self):
        env = gym.make('MountainCar-v0')
        agent = SimpleAgent()
        for episode in range(BOOTSTRAP_EPISODES):
            sys.stdout.write("Bootstrapping {} / {} \r".format(episode, BOOTSTRAP_EPISODES))
            sys.stdout.flush()
            ob = env.reset()
            obs = [ob]
            done = False
            total_reward = np.float32(0.0)
            while not done:
                action = agent.get_action(ob)
                ob_next, _, done, _ = env.step(action)
                obs.append(ob_next)
                total_reward = np.subtract(total_reward, np.float32(1.0))
                ob = ob_next

            for idx_next in range(1, len(obs)):
                ob = obs[idx_next-1]
                ob_next = obs[idx_next]
                output_next = self.get_output(np.array([ob_next]))
                self.sess.run(self.train_op, feed_dict={self.x: np.array([ob]), self.output_next: output_next})

            self.sess.run(self.train_op, feed_dict={self.x: np.array([obs[len(obs)-1]]),
                                                    self.output_next: np.array([[self.reward_to_output(total_reward)]])})
