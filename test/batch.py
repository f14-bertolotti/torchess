import unittest
import chess
import torch
import pysrc

class Suite(unittest.TestCase): 

    def test_0(self):

        moves = [pysrc.utils.parse_game(game) for game in pysrc.utils.games] * 10000
        lens  = torch.tensor([len(move) for move in moves]).to("cuda:0")
        moves = torch.nn.utils.rnn.pad_sequence(moves, batch_first=True)
        moves = torch.cat([moves, torch.zeros((moves.size(0),1,5), dtype=torch.int, device="cuda:0")], dim=1)

        board, player = pysrc.utils.chessboard2tensor(chess.Board())
        board, player = board.repeat(moves.size(0),1), player.repeat(moves.size(0))

        print(moves.shape, player.shape, board.shape)

        donestack, rewardstack, dones = [], [], None
        for i in range(moves.size(1)):
            rewards, dones = pysrc.step(board, moves[:,i], player, dones)
            donestack.append(dones.clone())
            rewardstack.append(rewards.clone())

        donestack = torch.stack(donestack)
        rewardstack = torch.stack(rewardstack)

        self.assertTrue(torch.equal(donestack.logical_not().int().sum(0), lens))
        self.assertTrue(rewardstack.sum(0).sum() == 0)


if __name__ == '__main__':
    unittest.main()
