import unittest
import chess
import torch

from pysrc.pawner import king
from pysrc.utils import str2chs, chs2pwn

def move(stringboard,turn,rights,mv):
    chessboard = str2chs(stringboard, turn, rights)
    torchboard,torchplayers = chs2pwn(chessboard)
    torchboard = torchboard.to("cuda:0")
    torchaction = torch.tensor([mv[1]], dtype=torch.int)
    torch_err = king(torchboard, torchaction.to("cuda:0"), torchplayers.to("cuda:0")).item()

    try:
        chess.Move.from_uci(mv[0])
        chessboard.push_uci(mv[0])
        chess_err = 0
    except Exception as e:
        chess_err = 1
    
    return torch_err, chess_err, torchboard[:,:64], chs2pwn(chessboard)[0][:,:64]


class Suite(unittest.TestCase): 

    def test_black_forward(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♚ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "", ("f5f6",[3,5,2,5,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_black_backward(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♚ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "", ("f5f4",[3,5,4,5,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_black_right(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♚ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "", ("f5g5",[3,5,3,6,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())


    def test_black_left(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♚ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "", ("f5e5",[3,5,3,4,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_white_forward(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♔ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.WHITE, "", ("f5f6",[3,5,2,5,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_white_backward(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♔ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.WHITE, "", ("f5f4",[3,5,4,5,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_white_right(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♔ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.WHITE, "", ("f5g5",[3,5,3,6,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())


    def test_white_left(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♔ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.WHITE, "", ("f5e5",[3,5,3,4,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

