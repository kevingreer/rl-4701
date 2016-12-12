import os
import random
import time

import numpy as np
import tensorflow as tf

from agents.ai_agent import AiAgent
from game import Game
from sts_test import shared_tester
import sys

HIDDEN1_PARTIAL_SIZE = 50
HIDDEN2_SIZE = 50
# FEATURE_SEGMENT_SIZES = [17, 208, 128]
FEATURE_SEGMENT_SIZES = [17, 240, 64]

SUMMARY_DIR = '../summaries/'
CHECKPOINT_DIR = '../checkpoints/'

EPISODES = 256
SUMMARY_INTERVAL = 200
CHECKPOINT_INTERVAL = 100
MOVES_PER_GAME = 12
LAMBDA = 0.7
BOOTSTRAP_SET_SIZE = 1000

PROB_THRESHOLD = 1.0  # 1 means regular TD-lambda


def connected_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='biases')
        return activation(tf.matmul(x, W) + b)


class TdModel(object):
    def __init__(self, sess, restore=False):
        self.sess = sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.lamda = tf.constant(LAMBDA)
        # self.alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step,
        #                         40000, 0.96, staircase=True), name='alpha')
        self.alpha = tf.constant(1.0)

        self.x_global = tf.placeholder('float', [1, FEATURE_SEGMENT_SIZES[0]], name='x_global')
        self.x_piece = tf.placeholder('float', [1, FEATURE_SEGMENT_SIZES[1]], name='x_piece')
        self.x_square = tf.placeholder('float', [1, FEATURE_SEGMENT_SIZES[2]], name='x_square')
        self.output_next = tf.placeholder('float', [1, 1], name='output_next')
        self.error = tf.placeholder('float', [1], name='error')

        self.output = self.construct_graph()
        delta_op = self.delta()
        self.train_op = self.training(delta_op)
        self.giraffe_train_op = self.training_batch()
        self.bootstrap_train_op = self.bootstrap_training()

        self.saver = tf.train.Saver(max_to_keep=1)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.train.SummaryWriter('{0}{1}'.format(SUMMARY_DIR, int(time.time()), sess.graph_def))

        self.sess.run(tf.global_variables_initializer())

        if restore:
            self.restore()

    def construct_graph(self):
        """
        Returns the operation to compute the network's output
        """
        hidden1_global = connected_layer(self.x_global, [FEATURE_SEGMENT_SIZES[0], HIDDEN1_PARTIAL_SIZE], tf.nn.relu, name='hidden1_global')
        hidden1_piece = connected_layer(self.x_piece, [FEATURE_SEGMENT_SIZES[1], HIDDEN1_PARTIAL_SIZE], tf.nn.relu, name='hidden1_piece')
        hidden1_square = connected_layer(self.x_square, [FEATURE_SEGMENT_SIZES[2], HIDDEN1_PARTIAL_SIZE], tf.nn.relu, name='hidden1_square')
        combined_hidden1 = tf.concat(1, [hidden1_global, hidden1_piece, hidden1_square])
        hidden2 = connected_layer(combined_hidden1, [HIDDEN1_PARTIAL_SIZE*3, HIDDEN2_SIZE], tf.nn.relu, name='hidden2')
        output = connected_layer(hidden2, [HIDDEN2_SIZE, 1], tf.tanh, name='output')
        return output

    def bootstrap_loss(self):
        return tf.reduce_mean(tf.square(self.output_next - self.output), name='bootstrap_loss')

    def bootstrap_training(self):
        loss = self.bootstrap_loss()
        optimizer = tf.train.AdadeltaOptimizer()
        bootstrap_global_step = tf.Variable(0, name='bootstrap_global_step', trainable=False)
        bootstrap_train_op = optimizer.minimize(loss, global_step=bootstrap_global_step)
        return bootstrap_train_op

    def delta(self):
        return tf.reduce_sum(self.output_next - self.output, name='delta')

    def training(self, delta_op):

        # It doesn't seem like AdaDeltaOptimizer (or any other SGD from TensorFlow) can easily do eligibility traces
        # so we need to do the updates ourselves
        trainable_vars = tf.trainable_variables()
        gradients = tf.gradients(self.output, trainable_vars)

        update_ops = [self.global_step.assign_add(1)]
        with tf.variable_scope('update_ops'):
            # There is a trace (and update) for every variable
            for gradient, variable in zip(gradients, trainable_vars):
                with tf.variable_scope('/e_trace'):
                    e_trace = tf.Variable(tf.zeros(gradient.get_shape()), trainable=False, name="e_trace")
                    e_trace_op = e_trace.assign((self.lamda * e_trace) + gradient)  # from sutton appendix

                update_op = variable.assign_add(self.alpha * delta_op * e_trace_op)
                update_ops.append(update_op)

        return tf.group(*update_ops, name='train')

    def training_batch(self):
        trainable_vars = tf.trainable_variables()
        gradients = tf.gradients(self.output, trainable_vars)

        update_ops = [self.global_step.assign_add(1)]
        with tf.variable_scope('update_ops'):
            for gradient, variable in zip(gradients, trainable_vars):
                update_op = variable.assign_add(gradient * self.error * self.alpha)
                update_ops.append(update_op)

        return tf.group(*update_ops, name='train')

    def calculate_error(self, sequence, players, player_idx):
        total_error = 0
        discount_factor = 1.0
        _, best_score = self.minimax(sequence[0].get_actions(), sequence[0], 1.0, sequence[0].player_ideal())
        for idx in range(1, len(sequence)-1):
            player_idx = (player_idx + 1) % 2
            _, best_score_next = self.minimax(sequence[idx].get_actions(), sequence[idx], 1.0, sequence[idx].player_ideal())
            delta = np.subtract(best_score_next, best_score)
            current_error = np.multiply(delta, discount_factor)
            discount_factor = np.multiply(discount_factor, LAMBDA)
            total_error = np.add(total_error, current_error)
        return total_error[0]

    def fill_feed_dict(self, xs, output_next=None, error=None):
        dict = {self.x_global: xs[0],
                self.x_piece: xs[1],
                self.x_square: xs[2]}
        if output_next is not None:
            dict[self.output_next] = output_next
        if error is not None:
            dict[self.error] = error
        return dict

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, xs):
        return self.sess.run(self.output, feed_dict={self.x_global: xs[0], self.x_piece: xs[1], self.x_square: xs[2]})

    def bootstrap(self):
        players = [AiAgent(self), AiAgent(self, material=True)]
        for i in range(BOOTSTRAP_SET_SIZE):
            sys.stdout.write("Bootstrapping {} / {} \r".format(i, BOOTSTRAP_SET_SIZE))
            sys.stdout.flush()
            game, half_turns = Game.random_position_from_next_training_game(players)
            truth = np.array([[game.stockfish_eval()]])
            # print truth
            xs = game.extract_features()
            self.sess.run([self.bootstrap_train_op, self.bootstrap_loss()],
                          feed_dict={self.x_global: xs[0], self.x_piece: xs[1], self.x_square: xs[2], self.output_next: truth})
        print "Finished bootstrapping"
        # shared_tester.test(players[0])


    def run_giraffe_training(self):
        players = [AiAgent(self, 1), AiAgent(self, -1)]

        for episode in range(EPISODES):
            game, half_turns = Game.random_position_from_next_training_game(players)
            random_action = random.choice(list(game.get_actions()))
            game.take_action(random_action)
            # game = Game.new(players)

            turn_count = half_turns + 1
            player_idx = turn_count % 2
            player_ideal = player_idx * -2 + 1
            game_sequence = [game.clone()]
            while not game.is_over():
                action = players[player_idx].get_action(game.get_actions(), game, ideal=player_ideal)
                game.take_action(action)
                game_sequence.append(game.clone())

                player_idx = (player_idx + 1) % 2
                player_ideal *= -1
                turn_count += 1

            winner = game.winner()
            print("Game %d/%d (Winner: %s) in %d turns" % (episode + 1, EPISODES, winner, turn_count))

            error = self.calculate_error(game_sequence, players, (half_turns + 1) % 2)
            # tf.scalar_summary('error', error)
            feed_dict = self.fill_feed_dict(game_sequence[0].extract_features(), error=error)
            _, global_step = self.sess.run([
                self.giraffe_train_op,
                self.global_step,
                # self.summary_op
            ], feed_dict=feed_dict)
            # self.summary_writer.add_summary(summaries, global_step=global_step)

            if episode % SUMMARY_INTERVAL:
                # TODO
                pass

            if ((episode + 1) % CHECKPOINT_INTERVAL == 0) or (episode + 1) == EPISODES:
                checkpoint_file = os.path.join(CHECKPOINT_DIR, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, global_step=self.global_step)
                shared_tester.test(players[0])

    def run_training(self):
        players = [AiAgent(self, 1), AiAgent(self, -1)]

        for episode in range(EPISODES):
            game, half_turns = Game.random_position_from_next_training_game(players)
            random_action = random.choice(list(game.get_actions()))
            game.take_action(random_action)
            # game = Game.new(players)

            player_idx = (half_turns + 1) % 2
            best_leaf, best_score = self.minimax(game.get_actions(), game, 1.0, game.player_ideal())
            features = best_leaf.extract_features()

            turn_count = half_turns
            while not game.is_over():
                player_idx = (player_idx + 1) % 2
                ideal = player_idx * -2 + 1
                action = players[player_idx].get_action(game.get_actions(), game, ideal=ideal)
                game.take_action(action)
                best_leaf, best_score = self.minimax(game.get_actions(), game, 1.0, game.player_ideal())
                features_next = best_leaf.extract_features()
                output_next = self.get_output(features_next)

                self.sess.run(self.train_op, feed_dict=self.fill_feed_dict(features, output_next=output_next))

                features = features_next
                turn_count += 1

            winner = game.winner()
            print("Game %d/%d (Winner: %s) in %d turns" % (episode+1, EPISODES, winner, turn_count))

            final_feed_dict = self.fill_feed_dict(features, output_next=np.array([[winner]]))
            _, global_step = self.sess.run([
                self.train_op,
                self.global_step,
            ], feed_dict=final_feed_dict)
            # self.summary_writer.add_summary(summaries, global_step=global_step)

            if episode % SUMMARY_INTERVAL:
                # TODO
                pass

            if ((episode + 1) % CHECKPOINT_INTERVAL == 0) or (episode + 1) == EPISODES:
                checkpoint_file = os.path.join(CHECKPOINT_DIR, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, global_step=self.global_step)
                shared_tester.test(players[0])

    def minimax(self, actions, game, current_prob, player_ideal):
        """
        Return the leaf game state of a minimax search
        """
        result = game.clone()
        if game.white_won():
            return result, 1
        elif game.black_won():
            return result, -1
        elif game.is_over():
            return result, 0

        if current_prob <= PROB_THRESHOLD:
            return result, self.get_output(game.extract_features())

        best_score = -3*player_ideal
        best_leaf = None

        for a in actions:
            # Our goal is player_ideal
            game_with_a = game.clone()
            game_with_a.take_action(a)
            # Now it's opponent's turn, they will choose game with score closest to -1*player_ideal
            moves = game_with_a.get_actions()
            subgame, subscore = self.minimax(moves, game_with_a, current_prob / len(actions), -1*player_ideal)
            # Subgame is what the opponent will do, we want closest to our own ideal
            if abs(player_ideal - subscore) < abs(player_ideal - best_score):
                best_score = subscore
                best_leaf = subgame

        return best_leaf, best_score

