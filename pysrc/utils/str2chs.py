from pysrc.utils import symbol2letter
import chess

def str2chs(string:str, turn, rights) -> chess.Board:
    board = chess.Board(None)
    for i,c in enumerate(string.replace(" ","").replace("\n","")): 
        row = i // 8
        col = i %  8
        board.set_piece_at((7-row) * 8 + col, symbol2letter(c))
    if rights: board.set_castling_fen(rights)
    board.turn = turn
    if rights and board.castling_xfen() != rights: raise ValueError("Invalid castling rights")
    return board


