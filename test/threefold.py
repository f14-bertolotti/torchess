import unittest
import chess
import pysrc
import torch

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
    
        chessboard = pysrc.utils.string2chessboard(stringboard, turn, rights)
        torchboard, players = pysrc.utils.chessboard2tensor(chessboard)
        for action in actions:
            reward,done = pysrc.step(torchboard, torch.tensor(action, dtype=torch.int, device="cuda:0").unsqueeze(0), players)
            players = 1-players
        self.assertEqual(reward[0,0].item(), -0.5)
        self.assertEqual(reward[0,1].item(), -0.5)
        self.assertTrue(done[0].item())



