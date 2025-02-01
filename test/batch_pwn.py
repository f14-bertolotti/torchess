import time
import chess
import torch
import pysrc


def main():
    batch_size = 1024

    actions = torch.randint(0, 8, (batch_size, 300, 5), device="cuda:0", dtype=torch.int)
    
    board, player = pysrc.utils.chessboard2tensor(chess.Board())
    board, player = board.repeat(actions.size(0),1), player.repeat(actions.size(0))

    pysrc.step(board, actions[:,0], player)  # (warmup) Run once before timing
    
    starttime = time.time()
    for i in range(actions.size(1)):
        pysrc.step(board, actions[:,i], player)
    endtime = time.time() - starttime
 
    print(f"Time taken: {endtime:.5f} seconds")
    print(f"Average time per step: {endtime/100:.5f} seconds")

   
    

if __name__ == '__main__':
    main()
