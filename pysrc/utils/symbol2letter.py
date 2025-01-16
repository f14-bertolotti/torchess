import chess

symbol2letter_ = {
    "⭘" : None,
    # black pieces
    "♟" : chess.Piece.from_symbol("p"),
    "♞" : chess.Piece.from_symbol("n"),
    "♝" : chess.Piece.from_symbol("b"),
    "♜" : chess.Piece.from_symbol("r"),
    "♛" : chess.Piece.from_symbol("q"),
    "♚" : chess.Piece.from_symbol("k"),
    # white pieces
    "♙" : chess.Piece.from_symbol("P"),
    "♘" : chess.Piece.from_symbol("N"),
    "♗" : chess.Piece.from_symbol("B"),
    "♖" : chess.Piece.from_symbol("R"),
    "♕" : chess.Piece.from_symbol("Q"),
    "♔" : chess.Piece.from_symbol("K")
}

def symbol2letter(symbol:str) -> chess.Piece:
    return symbol2letter_[symbol]
