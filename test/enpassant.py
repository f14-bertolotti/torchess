import unittest
import chess
import torch

from pysrc.pawner import doublepush, enpassant
from pysrc.utils import str2chs, chs2pwn

def move(stringboard,turn,rights,mv,prev):
    chessboard = str2chs(stringboard, turn, rights)
    chess.Move.from_uci(prev[0])
    chessboard.push_uci(prev[0])

    torchboard = chs2pwn(chessboard)
    if turn == chess.WHITE:
        torchboard[70:75,:] = torch.tensor(prev[1], dtype=torch.int).unsqueeze(1)
    else:
        torchboard[80:85,:] = torch.tensor(prev[1], dtype=torch.int).unsqueeze(1)
    torchboard = torchboard.to("cuda:0")
    torchaction = torch.tensor(mv[1], dtype=torch.int).unsqueeze(1)
    torch_err = doublepush(torchboard, torchaction.to("cuda:0")).item()
    torch_err = enpassant (torchboard, torchaction.to("cuda:0")).item()

    try:
        chess.Move.from_uci(mv[0])
        chessboard.push_uci(mv[0])
        chess_err = 0
    except Exception as e:
        chess_err = 1
    
    return torch_err, chess_err, torchboard[:64], chs2pwn(chessboard)[:64]


class Suite(unittest.TestCase): 

    def test_white(self):
        stringboard,turn,rights,action,prev = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♟ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ♙ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "", ("g5f6",[3,6,2,5,0]), ("f7f5",[1,5,3,5,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action,prev)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())

    def test_black(self):
        stringboard,turn,rights,action,prev = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ♟ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♙ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.WHITE, "", ("g4f3",[4,6,5,5,0]), ("f2f4",[6,5,4,5,0])

        torch_err, chess_err, torch_board, chess_board = move(stringboard,turn,rights,action,prev)
        self.assertTrue(torch_err == chess_err == 0)
        self.assertEqual(torch_board.tolist(), chess_board.tolist())


