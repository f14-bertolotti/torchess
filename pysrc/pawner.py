import torch
import cpawner

def kingside_castling(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.kingside_castling(board, action, result)
    return result

def queenside_castling(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.queenside_castling(board, action, result)
    return result

def promotion(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.promotion(board, action, result)
    return result

def pawn(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.pawn(board, action, result)
    return result

def knight(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.knight(board, action, result)
    return result

def king(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.king(board, action, result)
    return result

def rook(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.rook(board, action, result)
    return result

def bishop(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.bishop(board, action, result)
    return result

def queen(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.queen(board, action, result)
    return result

def doublepush(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.doublepush(board, action, result)
    return result

def enpassant(board:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
    result = torch.zeros(board.size(1), device=board.device, dtype=board.dtype)
    cpawner.enpassant(board, action, result)
    return result

def count_attacks(board:torch.Tensor):
    attacks = torch.zeros(1,64, dtype=board.dtype, device=board.device)
    cpawner.attacks(board, attacks)
    return attacks

def step(board:torch.Tensor, action:torch.Tensor, dones:torch.Tensor|None=None, rewards:torch.Tensor|None=None):
    dones   = torch.zeros((board.size(1)  ), device=board.device, dtype=torch.bool)  if dones   is None else dones
    rewards = torch.zeros((2,board.size(1)), device=board.device, dtype=torch.float) if rewards is None else rewards
    cpawner.step(board, action, rewards, dones)
    return rewards, dones
