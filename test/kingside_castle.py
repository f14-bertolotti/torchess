import pysrc.utils as utils
import unittest
import chess
import torch
import pysrc

reference_board = """
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
"""

def move(stringboard,turn,rights,move):
    chessboard = utils.string2chessboard(stringboard, turn, rights)
    torchboard,torchplayers = utils.chessboard2tensor(chessboard)
    torchboard = torchboard.to("cuda:0")
    torchaction = torch.tensor([[0,0,0,0,1]], dtype=torch.int)
    torch_err = pysrc.kingside_castling(torchboard, torchaction.to("cuda:0"), torchplayers.to("cuda:0")).item()

    try:
        chess.Move.from_uci(move)
        chessboard.push_uci(move)
        chess_err = 0
    except Exception as e:
        chess_err = 1
    
    return torch_err, chess_err, torchboard[:,:64], pysrc.utils.chessboard2tensor(chessboard)[0][:,:64]


class Suite(unittest.TestCase): 

    def test_white(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ⭘ ⭘ ♖
        """, chess.WHITE, "KQkq", "e1g1"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_black(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ⭘ ⭘ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        """, chess.BLACK, "kq", "e8g8"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_wrong_turn_1(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ⭘ ⭘ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
        """, chess.WHITE, "KQkq", "e8g8"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 1)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_wrong_turn_2(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ⭘ ⭘ ♖
        """, chess.BLACK, "KQkq", "e1g1"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 1)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_black_attacking(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ⭘ ⭘ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ⭘ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ♕ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "kq", "e8g8"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 1)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_white_attacking(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♜ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ⭘ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ⭘ ⭘ ♖
        """, chess.WHITE, "KQ", "e1g1"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 1)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())


    def test_black_check(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ⭘ ⭘ ♜
        ♟ ♟ ♟ ♟ ⭘ ♟ ⭘ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ♕ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "kq", "e8g8"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 1)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_white_check(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ♜ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ⭘ ⭘ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ⭘ ⭘ ♖
        """, chess.WHITE, "KQ", "e1g1"

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 1)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())


