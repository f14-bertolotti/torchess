import unittest
import chess
import pysrc
import torch


class Suite(unittest.TestCase): 

    def rungame(self, moves):

        chsboard = chess.Board()
        tchboard,tchplayer = pysrc.utils.chs2pwn(chsboard)

        # play game on both boards
        for i,move in enumerate(moves):

            # get actions in san and pwn format
            san = str(chsboard.parse_san(move))
            pwn = pysrc.utils.san2pwn(san, chsboard, chsboard.turn)

            # advance chess and pawner boards
            rew,_ = pysrc.step(tchboard,pwn.unsqueeze(0),tchplayer)
            chsboard.push_san(move)

            # compare chess and pawner boards
            self.assertTrue(torch.equal(pysrc.utils.chs2pwn(chsboard)[0][:,:64], tchboard[:,:64]))
            self.assertTrue(rew[0,0] == 0)
            self.assertTrue(rew[0,1] == 0)

        # check no other action is possible
        for pwn_action in pysrc.utils.pwn_actions():
            pwn = torch.tensor([[*pwn_action]], dtype=torch.int, device="cuda:0")
            rew, _ = pysrc.step(tchboard.clone(),pwn,tchplayer.clone())
            
            self.assertTrue(rew[0,tchplayer[0].item()].item() == -1)
            
    def test_1(self):
        moves = pysrc.utils.games[0].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_2(self):
        moves = pysrc.utils.games[1].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_3(self):
        moves = pysrc.utils.games[2].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_4(self):
        moves = pysrc.utils.games[3].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_5(self):
        moves = pysrc.utils.games[4].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_6(self):
        moves = pysrc.utils.games[5].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_7(self):
        moves = pysrc.utils.games[6].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_8(self):
        moves = pysrc.utils.games[7].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_9(self):
        moves = pysrc.utils.games[8].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_10(self):
        moves = pysrc.utils.games[9].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_11(self):
        moves = pysrc.utils.games[10].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_12(self):
        moves = pysrc.utils.games[11].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_13(self):
        moves = pysrc.utils.games[12].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_14(self):
        moves = pysrc.utils.games[13].split(" ")
        del moves[::3]
        self.rungame(moves)
    
    def test_15(self):
        moves = pysrc.utils.games[14].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_16(self):
        moves = pysrc.utils.games[15].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_17(self):
        moves = pysrc.utils.games[16].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_18(self):
        moves = pysrc.utils.games[17].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_19(self):
        moves = pysrc.utils.games[18].split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_20(self):
        moves = pysrc.utils.games[19].split(" ")
        del moves[::3]
        self.rungame(moves)
