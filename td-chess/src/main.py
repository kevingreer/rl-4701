import tensorflow as tf

from agents.ai_agent import AiAgent
from agents.random_agent import RandomAgent
from sts_test import shared_tester
from td_model import TdModel


def test(player):
    shared_tester.test(player)

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        model = TdModel(sess)
        players = [AiAgent(model), RandomAgent()]
        # model.bootstrap()
        # test(players[0])
        model.run_seq_training()
        # model.run_batch_training()
        # test(players[0])
