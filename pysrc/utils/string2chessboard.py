from pysrc.utils import symbol2letter
import chess

def string2chessboard(string:str, turn, rights) -> chess.Board:
    board = chess.Board(None)
    for i,c in enumerate(string.replace(" ","").replace("\n","")): 
        board.set_piece_at(i, symbol2letter(c))
    board.set_castling_fen(rights)
    board.turn = turn
    if board.castling_xfen() != rights: raise ValueError("Invalid castling rights")
    return board


