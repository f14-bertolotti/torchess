import unittest
import chess
import torch

from pysrc.pawner import step
from pysrc.utils import games, pwn_actions, san2pwn, chs2pwn

class Suite(unittest.TestCase): 

    def rungame(self, moves):

        chsboard = chess.Board()
        tchboard = chs2pwn(chsboard)

        # play game on both boards
        for i,move in enumerate(moves):

            # get actions in san and pwn format
            san = str(chsboard.parse_san(move))
            pwn = san2pwn(san, chsboard, chsboard.turn).unsqueeze(1)

            # advance chess and pawner boards
            rew,_ = step(tchboard,pwn)
            chsboard.push_san(move)

            # compare chess and pawner boards
            self.assertTrue(torch.equal(chs2pwn(chsboard)[:64], tchboard[:64]))
            self.assertTrue(rew[0,0] == 0)
            self.assertTrue(rew[1,0] == 0)

        # check no other action is possible
        for pwn_action in pwn_actions():
            pwn = torch.tensor([*pwn_action], dtype=torch.int, device="cuda:0").unsqueeze(-1)
            rew, _ = step(tchboard.clone(),pwn)
            
            self.assertTrue(rew[tchboard[96].item(),0].item() == -1)
            
    def test_1(self):
        moves = games[0].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_2(self):
        moves = games[1].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_3(self):
        moves = games[2].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_4(self):
        moves = games[3].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_5(self):
        moves = games[4].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_6(self):
        moves = games[5].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_7(self):
        moves = games[6].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_8(self):
        moves = games[7].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_9(self):
        moves = games[8].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_10(self):
        moves = games[9].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_11(self):
        moves = games[10].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_12(self):
        moves = games[11].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_13(self):
        moves = games[12].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_14(self):
        moves = games[13].split(" ")
        del moves[::3]
        self.rungame(moves)
    
    def test_15(self):
        moves = games[14].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_16(self):
        moves = games[15].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_17(self):
        moves = games[16].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_18(self):
        moves = games[17].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_19(self):
        moves = games[18].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_20(self):
        moves = games[19].split(" ")
        del moves[::3]
        self.rungame(moves)
