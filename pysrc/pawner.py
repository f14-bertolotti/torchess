import torch
import cpawner

def kingside_castling(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.kingside_castling(board, action, player, result)
    return result

def queenside_castling(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.queenside_castling(board, action, player, result)
    return result

def count_attacks(board:torch.Tensor, player:torch.Tensor):
    attacks = torch.zeros(1,64, dtype=board.dtype, device=board.device)
    cpawner.attacks(board, player, attacks)
    return attacks
