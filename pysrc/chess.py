import torch
import cpawner

def step(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor, dones:torch.Tensor|None=None, rewards:torch.Tensor|None=None) -> torch.Tensor:
    dones   = torch.zeros((player.size(0)  ), device=board.device, dtype=torch.bool)  if dones   is None else dones
    rewards = torch.zeros((player.size(0),2), device=board.device, dtype=torch.float) if rewards is None else rewards
    cpawner.step(board, action, player, rewards, dones)
    return rewards, dones
