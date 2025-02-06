import click
import time
import chess
import torch
from pysrc.utils import chs2pwn
from torchess import step

def main(batch_size = 4096):

    actions = torch.randint(0, 8, (1000, 5, batch_size), device="cuda:0", dtype=torch.int)
    
    board = chs2pwn(chess.Board())
    board = board.repeat(1,actions.size(0))

    reward = torch.empty(2, board.size(0), device="cuda:0", dtype=torch.float)
    dones  = torch.empty(board.size(0), device="cuda:0", dtype=torch.bool)
    step(board, actions[0], dones, reward)  # (warmup) Run once before timing
    
    starttime = time.time()
    for i in range(actions.size(0)):
        step(board, actions[i], dones, reward)
    endtime = time.time() - starttime
 
    print(f"Time taken: {endtime} seconds")
    print(f"Average time per step: {endtime/actions.size(0)} seconds")

    return endtime/actions.size(0)

@click.command()
@click.option("--batch-size", "batch_size", default=4096, help="Batch size for parallel execution")
def cli(batch_size):
    main(batch_size)

if __name__ == '__main__':
    cli()
