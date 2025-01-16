import unittest
import torch
import chess
import random
import pysrc

def get_random_board(seed):
    """ generates a random board """
    rng = random.Random(seed)
    board = chess.Board(None)
    pieces = list(chess.PIECE_TYPES)
    num_pieces = rng.randint(1, 32)
    for _ in range(num_pieces):
        square = rng.randint(0, 63)
        piece = rng.choice(pieces)
        color = rng.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(square, chess.Piece(piece, color))
    return board, rng.choice([0, 1])

def attacks_board(chess_board: chess.Board, color: chess.WHITE | chess.BLACK) -> torch.Tensor:
    """ given a chess board and a color returns a tensor with the number of attacks on each square """
    attacks = torch.zeros(1, 64, dtype=torch.int)
    attackers = []
    for square in chess.SQUARES: 
        attacks[0, square] += len(chess_board.attackers(color, square))
        attackers.append(chess_board.attackers(color, square))
    return attacks, attackers


def generate_tests():
    class DynamicTestCase(unittest.TestCase): pass

    # Dynamically add methods
    for i in range(1000):
        def test_template(self, seed=i):
            chess_board, color = get_random_board(seed)
            chess_attacks,attackers = attacks_board(chess_board, chess.WHITE if color == 0 else chess.BLACK)
        
            tensor_board,_ = pysrc.utils.chessboard2tensor(chess_board)
            tensor_board = tensor_board.to("cuda:0")
            pawner_attacks = pysrc.count_attacks(tensor_board, torch.tensor([color], device="cuda:0", dtype=torch.int))

            if not torch.all(chess_attacks.cpu() == pawner_attacks.cpu()):
                print(chess.Board())
                print()
                print(chess_board)
                print()
                print(pawner_attacks.view(8,8))
                print()
                print(chess_attacks.view(8,8))
                for i, attackers in enumerate(attackers):
                    print(f"{i}:\n{attackers}")
                print()
                print(tensor_board)

            self.assertEqual(chess_attacks.tolist(), pawner_attacks.tolist())

        # Add test method to class with a meaningful name
        setattr(DynamicTestCase, f'test_add_{i}', test_template)

    return DynamicTestCase

# Generate the test class
Suite = generate_tests()
