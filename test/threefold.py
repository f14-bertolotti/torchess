import unittest
import chess
import torch

from pysrc.pawner import step
from pysrc.utils import str2chs, chs2pwn

class Suite(unittest.TestCase): 

    def test_1(self):
        stringboard,turn,rights,actions = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ♝ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ♗ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        """, chess.BLACK, "", [
            [3,5,0,2,0],
            [5,2,4,1,0],
            [0,2,3,5,0],
            [4,1,5,2,0],
            [3,5,0,2,0],
            [5,2,4,1,0],
            [0,2,3,5,0],
            [4,1,5,2,0],
            [3,5,0,2,0],
            [5,2,4,1,0],
            [0,2,3,5,0],
            [4,1,5,2,0],
        ]
    
        chessboard = str2chs(stringboard, turn, rights)
        torchboard = chs2pwn(chessboard)
        for action in actions:
            reward,done = step(torchboard, torch.tensor(action, dtype=torch.int, device="cuda:0").unsqueeze(1))
        self.assertEqual(reward[0,0].item(), 0.5)
        self.assertEqual(reward[1,0].item(), 0.5)
        self.assertTrue(done[0].item())



