import random

PROB_THRESHOLD = 1

class AiAgent(object):
    def __init__(self, model, material=False):
        self.model = model
        self.name = "AI"
        self.nodes = 0
        self.material = material

    def get_action(self, moves, game, value=False, ideal=1):
        v_best = -2*ideal
        a_best = None

        for a in moves:
            game.take_action(a)
            # We want move that's closest to our ideal value
            _, best_score = self.minimax(game.get_actions(), game, 1.0, ideal)
            if abs(ideal - best_score) < abs(ideal - v_best):
                v_best = best_score
                a_best = a
            elif abs(ideal - best_score) == abs(ideal - v_best):
                a_best = random.choice([a_best, a])
            game.undo_action()
        return a_best if not value else v_best

    def minimax(self, actions, game, current_prob, player_ideal):
        """
        Return the leaf game state of a minimax search
        """
        self.nodes += 1
        result = game.clone()
        if game.white_won():
            return result, 1
        elif game.black_won():
            return result, -1
        elif game.is_over():
            return result, 0

        if current_prob <= PROB_THRESHOLD:
            value = game.material_eval() if self.material else self.model.get_output(game.extract_features())
            return result, value

        best_score = -3 * player_ideal
        best_leaf = None

        for a in actions:
            # Our goal is player_ideal
            game_with_a = game.clone()
            game_with_a.take_action(a)
            # Now it's opponent's turn, they will choose game with score closest to -1*player_ideal
            moves = game_with_a.get_actions()
            subgame, subscore = self.minimax(moves, game_with_a, current_prob / len(actions), -1 * player_ideal)
            # Subgame is what the opponent will do, we want closest to our own ideal
            if abs(player_ideal - subscore) < abs(player_ideal - best_score):
                best_score = subscore
                best_leaf = subgame

        return best_leaf, best_score
