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
        row = square // 8
        col = square %  8
        square_tgt = (7-row) * 8 + col
        attacks[0, square_tgt] += len(chess_board.attackers(color, square))
        attackers.append(chess_board.attackers(color, square))
    return attacks, attackers


def generate_tests():
    class DynamicTestCase(unittest.TestCase): 

        def test_base(self):
            chess_board = chess.Board()
            chess_attacks,attackers = attacks_board(chess_board, chess.BLACK)
            tensor_board,_ = pysrc.utils.chs2pwn(chess_board)
            tensor_board = tensor_board.to("cuda:0")
            pawner_attacks = pysrc.count_attacks(tensor_board, torch.tensor([0], device="cuda:0", dtype=torch.int))

            self.assertEqual(chess_attacks.tolist(), pawner_attacks.tolist())

        def single_black_pawn(self):
            stringboard,turn = """
            ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
            ⭘ ⭘ ⭘ ⭘ ⭘ ♟ ⭘ ⭘
            ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
            ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
            ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
            ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
            ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
            ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
            """, chess.BLACK

            chessboard = pysrc.utils.str2chs(stringboard,turn,"")
            tensorboard,tensorplayer = pysrc.utils.chs2pwn(chessboard)
            tensorboard = tensorboard.to("cuda:0")
            tensorplayer = tensorplayer.to("cuda:0")
            pawner_attacks = pysrc.count_attacks(tensorboard, tensorplayer)
            chess_attacks,attackers = attacks_board(chessboard, turn)

            self.assertEqual(chess_attacks.tolist(), pawner_attacks.tolist())


    # Dynamically add methods
    for i in range(100):
        def test_template(self, seed=i):
            chess_board, color = get_random_board(seed)
            chess_attacks,attackers = attacks_board(chess_board, chess.WHITE if color == 1 else chess.BLACK)
        
            tensor_board,_ = pysrc.utils.chs2pwn(chess_board)
            tensor_board = tensor_board.to("cuda:0")
            pawner_attacks = pysrc.count_attacks(tensor_board, torch.tensor([color], device="cuda:0", dtype=torch.int))

            self.assertEqual(chess_attacks.tolist(), pawner_attacks.tolist())

        # Add test method to class with a meaningful name
        setattr(DynamicTestCase, f'test_add_{i}', test_template)

    return DynamicTestCase

# Generate the test class
Suite = generate_tests()
