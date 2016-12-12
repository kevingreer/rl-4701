import chess
import chess.pgn
import random
import numpy as np
from chess import QUEEN, KING, PAWN, BISHOP, KNIGHT, ROOK, BLACK, WHITE
import math
from subprocess import Popen, PIPE

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NUM_PIECES = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
SLIDING_PIECES = [chess.BISHOP, chess.ROOK, chess.QUEEN]
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
PIECE_TO_NUM = {None: -1, chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4,
                chess.KING: 5}
RIGHT_EDGE = [7, 15, 23, 31, 39, 47, 55, 63]
LEFT_EDGE = [0, 8, 16, 24, 32, 40, 48, 56]

PGN_FILE = open('../data/CCRL-4040.[673112].pgn')

WK, WQ, WR, WN, WB, WP = 0, 1, 2, 3, 4, 5
BK, BQ, BR, BN, BB, BP = 8, 9, 10, 11, 12, 13

# most of the values in this file are taken from Stockfish
MAT = [
    [
        10000, # WK (we have a king score here for SEE only)
        2521, # WQ
        1270, # WR
        817, # WN
        836, # WB
        198, # WP

        0,
        0,

        10000, # BK
        2521, # BQ
        1270, # BR
        817, # BN
        836, # BB
        198 # BP
    ],
    [
        10000, # WK (we have a king score here for SEE only)
        2558, # WQ
        1278, # WR
        846, # WN
        857, # WB
        258, # WP

        0,
        0,

        10000, # BK
        2558, # BQ
        1278, # BR
        846, # BN
        857, # BB
        258 # BP
    ]
]

Q_PHASE_CONTRIBUTION = 4
R_PHASE_CONTRIBUTION = 2
B_PHASE_CONTRIBUTION = 1
N_PHASE_CONTRIBUTION = 1
P_PHASE_CONTRIBUTION = 0

MAX_PHASE = \
    Q_PHASE_CONTRIBUTION * 2 + \
    R_PHASE_CONTRIBUTION * 4 + \
    B_PHASE_CONTRIBUTION * 4 + \
    N_PHASE_CONTRIBUTION * 4 + \
    P_PHASE_CONTRIBUTION * 16


def scale_phase(opening_score, endgame_score, phase):
    diff = opening_score - endgame_score
    return endgame_score + diff * float(phase) / MAX_PHASE


class Game:
    def __init__(self, players, board=None):
        if not board:
            self.board = chess.Board()
        else:
            self.board = board
        self.players = players
        self.player_numbers = [0, 1]

    @staticmethod
    def new(players):
        game = Game(players)
        game.reset()
        return game

    def play(self):
        player_num = 0
        while not self.board.is_game_over():
            self.take_turn(self.players[player_num])
            player_num = (player_num + 1) % 2
        self.draw()
        return self.board.result()

    def take_turn(self, player):
        moves = self.get_actions()
        move = player.get_action(moves, self) if moves else None

        if move:
            self.take_action(move)
        else:
            self.board.push(chess.Move.null())

    def get_move(self, player, ideal=None):
        moves = self.get_actions()
        return player.get_action(moves, self, ideal=ideal) if moves else None

    def opponent(self, player):
        if player == self.players[0]:
            return self.players[1]
        elif player == self.players[1]:
            return self.players[0]
        return None

    # Not sure if this works
    def clone(self):
        return Game(self.players, board=self.board.copy())

    def take_action(self, move):
        self.board.push(move)

    def undo_action(self):
        self.board.pop()

    def get_actions(self):
        return self.board.legal_moves

    def white_won(self):
        return self.winner() == 1

    def black_won(self):
        return self.winner() == -1

    def is_won(self, player):
        if self.player_numbers[0] == player:
            desired_result = "1-0"
        else:
            desired_result = "0-1"
        return self.board.is_game_over() and self.board.result() == desired_result

    def is_lost(self, player):
        if self.player_numbers[0] == player:
            desired_result = "0-1"
        else:
            desired_result = "1-0"
        return self.board.is_game_over() and self.board.result() == desired_result

    def reverse(self):
        pass

    def reset(self):
        self.board.reset()

    def winner(self):
        result = self.board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        elif result == "1/2-1/2":
            return 0
        return None

    def is_over(self):
        return self.board.is_game_over()

    def draw(self):
        print self.board

    def square_on_bottom_edge(self, index):
        return index >= 0 and index <= 7

    def square_on_top_edge(self, index):
        return index >= 56 and index <= 63

    def piece_on_square(self, square):
        if self.board.piece_at(square) is None:
            return False
        else:
            return True

    def should_stop_moving(self, direction, board_index):
        if direction == 'UPLEFT':
            return board_index in LEFT_EDGE or self.square_on_top_edge(board_index)
        if direction == 'UP':
            return self.square_on_top_edge(board_index)
        if direction == 'UPRIGHT':
            return board_index in RIGHT_EDGE or self.square_on_top_edge(board_index)
        if direction == 'RIGHT':
            return board_index in RIGHT_EDGE
        if direction == 'DOWNRIGHT':
            return board_index in RIGHT_EDGE or self.square_on_bottom_edge(board_index)
        if direction == 'DOWN':
            return self.square_on_bottom_edge(board_index)
        if direction == 'DOWNLEFT':
            return board_index in LEFT_EDGE or self.square_on_bottom_edge(board_index)
        if direction == 'LEFT':
            return board_index in LEFT_EDGE

    def get_sliding_spaces(self, piece, square):
        i = 0
        sliding_spaces = [-1] * 8
        directions = ['UPLEFT', 'UP', 'UPRIGHT', 'RIGHT', 'DOWNRIGHT', 'DOWN', 'DOWNLEFT', 'LEFT']
        direction_to_index = {'UPLEFT': 7, 'UP': 8, 'UPRIGHT': 9, 'RIGHT': 1, 'DOWNRIGHT': -7,
                                'DOWN': -8, 'DOWNLEFT': -9, 'LEFT': -1}
        piece_directions = {chess.BISHOP: ['UPLEFT', 'UPRIGHT', 'DOWNRIGHT', 'DOWNLEFT'],
                            chess.ROOK: ['UP', 'RIGHT', 'DOWN', 'LEFT'],
                            chess.QUEEN: ['UPLEFT', 'UP', 'UPRIGHT', 'RIGHT', 'DOWNRIGHT', 'DOWN', 'DOWNLEFT', 'LEFT']}
        for direction in directions:
            board_index = square
            counter = 0
            if direction in piece_directions[piece]:
                while not self.should_stop_moving(direction, board_index):
                    board_index += direction_to_index[direction]
                    if self.piece_on_square(board_index):
                        break
                    counter +=1
                sliding_spaces[i] = counter
            i += 1
        return sliding_spaces

    def get_lowest_attacker_of_color(self, color, square):
        attackers = []
        if color == chess.WHITE:
            attackers.extend(list(self.board.attackers(chess.WHITE, square)))
        else:
            attackers.extend(list(self.board.attackers(chess.BLACK, square)))
        lowest = 101
        lowest_piece = None
        for attacker_square in attackers:
            piece = self.board.piece_at(attacker_square)
            if PIECE_VALUES[piece.piece_type] < lowest:
                lowest = PIECE_VALUES[piece.piece_type]
                lowest_piece = piece.piece_type
        return PIECE_TO_NUM[lowest_piece]

    def get_lowest_attacker_square(self, square):
        attackers = []
        attackers.extend(list(self.board.attackers(chess.WHITE, square)))
        attackers.extend(list(self.board.attackers(chess.BLACK, square)))
        lowest = 101
        lowest_piece = None
        for attacker_square in attackers:
            piece = self.board.piece_at(attacker_square)
            if PIECE_VALUES[piece.piece_type] < lowest:
                lowest = PIECE_VALUES[piece.piece_type]
                lowest_piece = piece.piece_type
        return PIECE_TO_NUM[lowest_piece]

    def get_lowest_defender_of_color(self, color, square):
        defenders = []
        if color == chess.WHITE:
            # make a copy of the board but put a black piece at square
            # see which white pieces can attack this square
            # get the lowest-valued piece out of these white pieces
            copy = self.board.copy()
            copy.set_piece_at(square, chess.Piece(chess.PAWN, chess.BLACK))
            defenders.extend(list(copy.attackers(chess.WHITE, square)))
            lowest = 101
            lowest_piece = None
            for defender_square in defenders:
                piece = self.board.piece_at(defender_square)
                if PIECE_VALUES[piece.piece_type] < lowest:
                    lowest = PIECE_VALUES[piece.piece_type]
                    lowest_piece = piece.piece_type
            return PIECE_TO_NUM[lowest_piece]
        else:
            copy = self.board.copy()
            copy.set_piece_at(square, chess.Piece(chess.PAWN, chess.WHITE))
            defenders.extend(list(copy.attackers(chess.BLACK, square)))
            lowest = 101
            lowest_piece = None
            for defender_square in defenders:
                piece = self.board.piece_at(defender_square)
                if PIECE_VALUES[piece.piece_type] < lowest:
                    lowest = PIECE_VALUES[piece.piece_type]
                    lowest_piece = piece.piece_type
            return PIECE_TO_NUM[lowest_piece]

    @staticmethod
    def next_training_game():
        return chess.pgn.read_game(PGN_FILE)

    @staticmethod
    def _game_node_turns(game_node):
        half_turns = 1
        while len(game_node.variations) > 0:
            half_turns += 1
            game_node = game_node.variations[0]
        return half_turns

    @staticmethod
    def random_position_from_next_training_game(players):
        next_game = Game.next_training_game()
        half_turns = random.randint(1, Game._game_node_turns(next_game)-3)  # need to be able to apply a random move
        for _ in range(half_turns):
            next_game = next_game.variations[0]
        return Game(players, board=next_game.board()), half_turns

    def player_ideal(self):
        return 1 if self.board.turn else -1

    def extract_features(self):
        global_feats = []
        piece_feats = []
        square_feats = []

        global_feats.append(1) if self.board.turn else global_feats.append(0)
        global_feats.append(1) if self.board.has_kingside_castling_rights(chess.WHITE) else global_feats.append(0)
        global_feats.append(1) if self.board.has_queenside_castling_rights(chess.WHITE) else global_feats.append(0)
        global_feats.append(1) if self.board.has_kingside_castling_rights(chess.BLACK) else global_feats.append(0)
        global_feats.append(1) if self.board.has_queenside_castling_rights(chess.BLACK) else global_feats.append(0)

        for i in range(2):
            for piece in PIECE_TYPES:
                if i == 0:
                    squares = self.board.pieces(piece, chess.WHITE)
                else:
                    squares = self.board.pieces(piece, chess.BLACK)
                global_feats.append(len(squares))
                if len(squares) == 0:
                    for i in range(NUM_PIECES[piece]):
                        piece_feats.append(0)
                        piece_feats.append(-1)
                        piece_feats.append(-1)
                        # Need to add filler stuff for lowest attacker and defender sliding_spaces
                        piece_feats.append(-1)
                        piece_feats.append(-1)
                        if piece in SLIDING_PIECES:
                            for _ in range(8):
                                piece_feats.append(-1)
                else:
                    index = 0
                    list_of_squares = list(squares)
                    for i in range(NUM_PIECES[piece]):
                        if index < len(list_of_squares):
                            piece_feats.append(1)
                            piece_feats.append(list_of_squares[index] % 8)
                            piece_feats.append(list_of_squares[index] / 8)
                            if i == 0:
                                piece_feats.append(self.get_lowest_attacker_of_color(chess.BLACK,
                                                                                     list_of_squares[index]))
                            else:
                                piece_feats.append(self.get_lowest_attacker_of_color(chess.WHITE,
                                                                                     list_of_squares[index]))
                            if i == 0:
                                piece_feats.append(self.get_lowest_defender_of_color(chess.WHITE,
                                                                                     list_of_squares[index]))
                            else:
                                piece_feats.append(self.get_lowest_defender_of_color(chess.BLACK,
                                                                                     list_of_squares[index]))
                            if piece in SLIDING_PIECES:
                                piece_feats.extend(self.get_sliding_spaces(piece, list_of_squares[index]))
                            index += 1
                        else:
                            piece_feats.append(0)
                            piece_feats.append(-1)
                            piece_feats.append(-1)
                            # Need to add filler stuff for lowest attacker and defender sliding_spaces
                            piece_feats.append(-1)
                            piece_feats.append(-1)
                            if piece in SLIDING_PIECES:
                                for _ in range(8):
                                    piece_feats.append(-1)

        for square in chess.SQUARES:
            square_feats.append(self.get_lowest_attacker_square(square))
        #     if self.board.piece_at(square) is not None:
        #         if self.board.piece_at(square).color == chess.WHITE:
        #             square_feats.append(self.get_lowest_defender_of_color(chess.WHITE, square))
        #         else:
        #             square_feats.append(self.get_lowest_defender_of_color(chess.BLACK, square))
        #     else:
        #         square_feats.append(-1)

        return np.array([global_feats]), np.array([piece_feats]), np.array([square_feats])


    def material_eval(self):
        piece_count = lambda piece_type, color: np.float32(len(self.board.pieces(piece_type, color)))

        wq_count = piece_count(QUEEN, WHITE)
        wr_count = piece_count(ROOK, WHITE)
        wb_count = piece_count(BISHOP, WHITE)
        wn_count = piece_count(KNIGHT, WHITE)
        wp_count = piece_count(PAWN, WHITE)

        bq_count = piece_count(QUEEN, BLACK)
        br_count = piece_count(ROOK, BLACK)
        bb_count = piece_count(BISHOP, BLACK)
        bn_count = piece_count(KNIGHT, BLACK)
        bp_count = piece_count(PAWN, BLACK)

        phase = np.sum([
            np.multiply(wq_count, Q_PHASE_CONTRIBUTION),
            np.multiply(bq_count, Q_PHASE_CONTRIBUTION),
            np.multiply(wr_count, R_PHASE_CONTRIBUTION),
            np.multiply(br_count, R_PHASE_CONTRIBUTION),
            np.multiply(wb_count, B_PHASE_CONTRIBUTION),
            np.multiply(bb_count, B_PHASE_CONTRIBUTION),
            np.multiply(wn_count, N_PHASE_CONTRIBUTION),
            np.multiply(bn_count, N_PHASE_CONTRIBUTION),
            np.multiply(wp_count, P_PHASE_CONTRIBUTION),
            np.multiply(bp_count, P_PHASE_CONTRIBUTION)])

        phase = np.minimum(phase, MAX_PHASE)

        white_contrib = np.sum([
            np.multiply(wq_count, scale_phase(MAT[0][WQ], MAT[1][WQ], phase)),
            np.multiply(wr_count, scale_phase(MAT[0][WR], MAT[1][WR], phase)),
            np.multiply(wb_count, scale_phase(MAT[0][WB], MAT[1][WB], phase)),
            np.multiply(wn_count, scale_phase(MAT[0][WN], MAT[1][WN], phase)),
            np.multiply(wp_count, scale_phase(MAT[0][WP], MAT[1][WP], phase)),
        ])

        black_contrib = np.sum([
            np.multiply(bq_count, scale_phase(MAT[0][WQ], MAT[1][WQ], phase)),
            np.multiply(br_count, scale_phase(MAT[0][WR], MAT[1][WR], phase)),
            np.multiply(bb_count, scale_phase(MAT[0][WB], MAT[1][WB], phase)),
            np.multiply(bn_count, scale_phase(MAT[0][WN], MAT[1][WN], phase)),
            np.multiply(bp_count, scale_phase(MAT[0][WP], MAT[1][WP], phase)),
        ])

        ret = np.subtract(white_contrib, black_contrib)

        return np.tanh(np.multiply(0.001,ret))

    def stockfish_eval(self):
        p = Popen(['./stockfish', "position fen {}".format(self.board.fen()), 'eval', 'quit'], stdout=PIPE)
        result = p.stdout.readlines()[0].rstrip('\n')
        return np.float32(result)