import chess


class HumanAgent(object):
    def __init__(self):
        self.name = 'Human'

    def get_action(self, moves, game):
        if not moves:
            raw_input('No moves for you... (hit enter)')
            return None

        while True:
            mv = raw_input('Enter a move in UCI form or type "board" to see the board\n')
            if mv.lower() == 'board':
                print game.board
            elif mv.lower() == 'moves':
                print moves
            else:
                if len(mv) != 4 and len(mv) != 5:
                    print "Bad Format"
                    continue
                move = chess.Move.from_uci(mv)
                if move in moves:
                    return move
                else:
                    print "You can't play that move"