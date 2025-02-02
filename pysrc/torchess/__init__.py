import torch
import cpawner

baseboard = torch.tensor([[
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
]], device='cuda:0', dtype=torch.int32)
baseplayer = torch.tensor([0], device='cuda:0', dtype=torch.int32)

def step(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor, dones:torch.Tensor, rewards:torch.Tensor) -> None:
    """
    Step the environment forward by one in place.
    board:   (batch, 100) - the current board state
    action:  (batch, 5)   - the action to be taken
    player:  (batch, 2)   - the player's turn
    dones:   (batch)      - the done flag
    rewards: (batch, 2)   - the rewards
    """
    cpawner.step(board, action, player, rewards, dones)
    return None

def init(envs:int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize the environment
    envs: (batch) - the number of environments to initialize
    """
    return baseboard.repeat(envs,1), baseplayer.repeat(envs)

def reset(boards:torch.Tensor, players:torch.Tensor, mask:torch.Tensor|None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reset the environment
    envs: (batch) - the number of environments to reset
    """
    if mask is None: return init(boards.size(0))
    boards [mask] = baseboard
    players[mask] = baseplayer
    return boards, players

