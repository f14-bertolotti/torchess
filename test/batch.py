import unittest
import chess
import torch

from pysrc.pawner import step
from pysrc.utils import parse_game, games, chs2pwn

class Suite(unittest.TestCase): 

    def test_0(self):

        moves = [parse_game(game) for game in games] * 10
        lens  = torch.tensor([len(move) for move in moves]).to("cuda:0")
        moves = torch.nn.utils.rnn.pad_sequence(moves, batch_first=True)
        moves = torch.cat([moves, torch.zeros((moves.size(0),1,5), dtype=torch.int, device="cuda:0")], dim=1).permute(2,1,0)

        board, player = chs2pwn(chess.Board())
        board, player = board.repeat(1,moves.size(2)), player.repeat(moves.size(2))

        donestack, rewardstack, dones = [], [], None
        for i in range(moves.size(1)):
            rewards, dones = step(board, moves[:,i], player, dones)
            donestack.append(dones.clone())
            rewardstack.append(rewards.clone())

        donestack = torch.stack(donestack)
        rewardstack = torch.stack(rewardstack)

        self.assertTrue(torch.equal(donestack.logical_not().int().sum(0), lens))
        self.assertTrue(rewardstack.sum(0).sum() == 0)

    def test_1(self):
        from pysrc.torchess import step, init
        
        actions = [parse_game(games[0])] * 128
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True)
        actions = torch.cat([actions, torch.zeros((actions.size(0),1,5), dtype=torch.int, device="cuda:0")], dim=1).permute(2,1,0)

        boards,players = init(128)
        dones, rewards = torch.zeros(100, dtype=torch.bool, device="cuda:0"), torch.zeros(2,100,dtype=torch.float, device="cuda:0")

        for i in range(actions.size(1)):
            step(boards, actions[:,i], players, dones, rewards)

        self.assertTrue(dones.all())
        self.assertTrue((rewards[0,:] == -1).all())
        self.assertTrue((rewards[1,:] == +1).all())


if __name__ == '__main__':
    unittest.main()
