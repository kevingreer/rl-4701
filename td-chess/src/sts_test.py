import os
from game import Game
import chess
from agents.random_agent import RandomAgent
import re
import sys

DIRECTORY = '../STS'
NUM_SUITES = 15

class StsTester:
    def __init__(self):
        self.test_suites = self.parse_epds()

    def parse_epds(self):
        test_suites = [None] * NUM_SUITES
        for filename in os.listdir(DIRECTORY):
            if 'STS' not in filename:
                continue
            dump = open(os.path.join(DIRECTORY, filename)).read()
            epd_strings = dump.split('\n')
            index = int(re.search(r'\d+', filename).group()) - 1
            test_suites[index] = epd_strings
        return test_suites

    def test(self, player):
        game = Game([player, RandomAgent()])
        test_counter = 1
        score = 0
        for test_suite in self.test_suites:
            sys.stdout.write('Running STS test suite: {} / 15 \n'.format(test_counter))
            sys.stdout.flush()
            # count = 0
            for epd_string in test_suite:
                if epd_string == '':
                    continue
                # print epd_string
                (game.board, operations) = chess.Board.from_epd(epd_string)
                side_to_move = epd_string.split(' ')[1]
                # print self.game.board
                # Make game take a turn with the test player
                move = game.get_move(player, ideal=1 if side_to_move == 'w' else -1)
                # print "******{}******".format(count)
                # count += 1
                # Get points based on move made or 0 if the move is not in operations
                # print move
                san_move = game.board.san(move)
                if 'c0' in operations:
                    score_list = operations['c0']
                    score_list = score_list.split(', ')
                    this_score = 0
                    for item in score_list:
                        # make this a move object
                        splits = item.split('=')
                        move_to_check = splits[0]
                        if san_move == move_to_check:
                            this_score = int(splits[1])
                    score += this_score
                else:
                    if operations['bm'][0].uci() == move.uci():
                        score += 10
            test_counter += 1
            print score
        print "Finished running test suite: score {} / 15000".format(score)
        return score

shared_tester = StsTester()