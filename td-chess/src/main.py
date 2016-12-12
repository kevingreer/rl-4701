from game import Game
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.ai_agent import AiAgent
from td_model import TdModel
from sts_test import shared_tester
import tensorflow as tf

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        model = TdModel(sess)
        # model.bootstrap()
        players = [AiAgent(model), RandomAgent()]
        shared_tester.test(players[0])
        # wins = 0.0
        # for i in range(100):
        #     game = Game.new(players)
        #     game.play()
        #     wins = wins + 1 if game.white_won() else wins
        #     print wins / (i+1)
        #     exit()
        model.run_giraffe_training()
        # model.run_training()

