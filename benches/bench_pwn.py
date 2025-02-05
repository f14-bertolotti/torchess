import time
import chess
import torch
from pysrc.utils import chs2pwn
from torchess import step


def main():
    batch_size = 4096

    actions = torch.randint(0, 8, (1000, 5, batch_size), device="cuda:0", dtype=torch.int)
    
    board, player = chs2pwn(chess.Board())
    board, player = board.repeat(1,actions.size(0)), player.repeat(actions.size(0))

    reward = torch.empty(2, board.size(0), device="cuda:0", dtype=torch.float)
    dones  = torch.empty(board.size(0), device="cuda:0", dtype=torch.bool)
    step(board, actions[0], player, dones, reward)  # (warmup) Run once before timing
    
    starttime = time.time()
    for i in range(actions.size(0)):
        step(board, actions[i], player, dones, reward)
    endtime = time.time() - starttime
 
    print(f"Time taken: {endtime:.5f} seconds")
    print(f"Average time per step: {endtime/100:.5f} seconds")

if __name__ == '__main__':
    main()
