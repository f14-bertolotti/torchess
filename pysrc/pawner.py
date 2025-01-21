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

def pawn_move(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.pawn_move(board, action, player, result)
    return result

def knight_move(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.knight_move(board, action, player, result)
    return result

def king_move(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.king_move(board, action, player, result)
    return result

def rook_move(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.rook_move(board, action, player, result)
    return result

def bishop_move(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.bishop_move(board, action, player, result)
    return result

def queen_move(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.queen_move(board, action, player, result)
    return result

def pawn_double(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.pawn_double(board, action, player, result)
    return result

def en_passant(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(player, device=board.device, dtype=board.dtype)
    cpawner.pawn_en_passant(board, action, player, result)
    return result

def count_attacks(board:torch.Tensor, player:torch.Tensor):
    attacks = torch.zeros(1,64, dtype=board.dtype, device=board.device)
    cpawner.attacks(board, player, attacks)
    return attacks

def step(board:torch.Tensor, action:torch.Tensor, player:torch.Tensor):
    dones   = torch.zeros(player.size(0), device=board.device, dtype=torch.bool)
    rewards = torch.zeros(player.size(0),2, device=board.device, dtype=torch.float)
    cpawner.step(board, action, player, rewards, dones)
    return rewards, dones
