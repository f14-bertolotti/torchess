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

def promotion(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.promotion(board, action, player, result)
    return result

def pawn(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.pawn(board, action, player, result)
    return result

def knight(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.knight(board, action, player, result)
    return result

def king(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.king(board, action, player, result)
    return result

def rook(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.rook(board, action, player, result)
    return result

def bishop(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.bishop(board, action, player, result)
    return result

def queen(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.queen(board, action, player, result)
    return result

def doublepush(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.doublepush(board, action, player, result)
    return result

def enpassant(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.enpassant(board, action, player, result)
    return result

def count_attacks(board:torch.Tensor, player:torch.Tensor):
    attacks = torch.zeros(1,64, dtype=board.dtype, device=board.device)
    cpawner.attacks(board, player, attacks)
    return attacks

def step(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor, dones:torch.Tensor|None=None):
    dones   = torch.zeros((player.size(0)  ), device=board.device, dtype=torch.bool)  if dones   is None else dones
    rewards = torch.zeros((player.size(0),2), device=board.device, dtype=torch.float) if rewards is None else rewards
    cpawner.step(board, action, player, rewards, dones)
    return rewards, dones
