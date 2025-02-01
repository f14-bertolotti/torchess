import pysrc
import torch
import chess

def parse_game(game:str):

    board = chess.Board()

    moves = game.split(" ")
    del moves[::3]
    
    parsed_moves = []
    for move in moves:
        move = board.parse_san(move)
        parsed_moves.append(pysrc.utils.san2pwn(str(move), board, board.turn))
        board.push(move)
    parsed_moves = torch.stack(parsed_moves)

    return parsed_moves

