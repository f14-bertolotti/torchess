import unittest
import chess
import torch

from pysrc.pawner import step
from pysrc.utils import str2chs, chs2pwn

def move(stringboard,turn,rights,mv):
    chessboard = str2chs(stringboard, turn, rights)
    torchboard = chs2pwn(chessboard)
    torchboard = torchboard.to("cuda:0")
    torchaction = torch.tensor(mv[1], dtype=torch.int).unsqueeze(1)

    torch_rew, torch_win = step(torchboard, torchaction.to("cuda:0"))

    try:
        chess.Move.from_uci(mv[0])
        chessboard.push_uci(mv[0])
        chess_err = 0
    except Exception as e:
        chess_err = 1
    
    return torch_rew, torch_win.item(), chess_err, torchboard[:,:64], chs2pwn(chessboard)[0][:,:64]


class Suite(unittest.TestCase): 

    def test_black(self):
        stringboard,turn,rights,action = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♟ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "", ("f7f6",[1,5,2,5,0])

        torch_rew, torch_win, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(chess_err == 0)
        self.assertTrue(torch_rew[0,0] == 0)
        self.assertTrue(torch_rew[1,0] == 0)
        self.assertTrue(torch_win == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())


    def test_1(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
        """, chess.WHITE, "KQkq", ("a2b7",[6,0,1,1,0])

        torch_rew, torch_win, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(chess_err == 1)
        self.assertTrue(torch_rew[0,0] == -1)
        self.assertTrue(torch_rew[0,1] == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())



    def test_2(self):
        stringboard,turn,rights,action = """
        ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
        """, chess.WHITE, "KQkq", ("a2a4",[6,0,4,0,0])

        torch_rew, torch_win, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action)
        self.assertTrue(chess_err == 0)
        self.assertTrue(torch_rew[0,0] == 0)
        self.assertTrue(torch_rew[0,1] == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

  
