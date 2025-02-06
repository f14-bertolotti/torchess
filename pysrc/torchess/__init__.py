import torch
import cpawner

baseboard = torch.tensor([
        10 , 8 , 9 , 11 , 12 , 9 , 8 , 10 ,
        7  , 7 , 7 , 7  , 7  , 7 , 7 , 7  ,
        0  , 0 , 0 , 0  , 0  , 0 , 0 , 0  ,
        0  , 0 , 0 , 0  , 0  , 0 , 0 , 0  ,
        0  , 0 , 0 , 0  , 0  , 0 , 0 , 0  ,
        0  , 0 , 0 , 0  , 0  , 0 , 0 , 0  ,
        1  , 1 , 1 , 1  , 1  , 1 , 1 , 1  ,
        4  , 2 , 3 , 5  , 6  , 3 , 2 , 4  ,
        0  , 0 , 0 , 0  , 0  , 0 , 0 , 0  ,
        0  , 0 , 0 , 0  , 0  , 0 , 0 , 0  ,
        0  , 0 , 0 , 0  , 0  , 0 , 0 , 0  ,
        0  , 0 , 7 , 4  , 0  , 4 , 0 , 0  ,
        0  , 0 , 0 , 0
], device='cuda:0', dtype=torch.int32).unsqueeze(1)

def step(board:torch.Tensor, action:torch.Tensor, dones:torch.Tensor, rewards:torch.Tensor) -> None:
    """
    Step the environment forward by one in place.
    board:   (100,batch)  - the current board state
    action:  (batch, 5)   - the action to be taken
    dones:   (batch)      - the done flag
    rewards: (batch, 2)   - the rewards
    """
    cpawner.step(board, action, rewards, dones)
    return None

def init(envs:int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize the environment
    envs: (batch) - the number of environments to initialize
    """
    return baseboard.repeat(1,envs)

def reset(boards:torch.Tensor, mask:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reset the environment in place
    boards: (100,batch) - the current board state
    mask:   (batch)      - the mask of environments to reset
    """
    boards [:,mask] = baseboard
    return None

