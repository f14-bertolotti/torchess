import torch
import chess

KING_POSITION = 90
CASTLING_RIGHTS = 64

def chs2pwn(chessboard: chess.Board):
    """ given a chess boards returns a torch tensor representation of it """

    torch_board = torch.zeros(1, 100, dtype=torch.int)
    players = torch.tensor([0 if chessboard.turn == chess.WHITE else 1], dtype=torch.int)
    for square, piece in chessboard.piece_map().items():
        row = square // 8
        col = square % 8
        square = row * 8 + 7 - col
        color = piece.color if piece is not None else None
        piece_type = piece.piece_type if piece is not None else None
        match (color, piece_type):
            case (None       , None         ): torch_board[0, 63-square] = 0
            case (chess.WHITE, chess.PAWN   ): torch_board[0, 63-square] = 1
            case (chess.WHITE, chess.KNIGHT ): torch_board[0, 63-square] = 2
            case (chess.WHITE, chess.BISHOP ): torch_board[0, 63-square] = 3
            case (chess.WHITE, chess.ROOK   ): torch_board[0, 63-square] = 4
            case (chess.WHITE, chess.QUEEN  ): torch_board[0, 63-square] = 5
            case (chess.WHITE, chess.KING   ): torch_board[0, 63-square] = 6; torch_board[0, KING_POSITION] = 7-row; torch_board[0, KING_POSITION+1] = col
            case (chess.BLACK, chess.PAWN   ): torch_board[0, 63-square] = 7
            case (chess.BLACK, chess.KNIGHT ): torch_board[0, 63-square] = 8
            case (chess.BLACK, chess.BISHOP ): torch_board[0, 63-square] = 9
            case (chess.BLACK, chess.ROOK   ): torch_board[0, 63-square] = 10
            case (chess.BLACK, chess.QUEEN  ): torch_board[0, 63-square] = 11
            case (chess.BLACK, chess.KING   ): torch_board[0, 63-square] = 12; torch_board[0, KING_POSITION+2] = 7-row; torch_board[0, KING_POSITION+3] = col
            case _ : raise ValueError(f"Invalid piece: {piece}")

    torch_board[0, CASTLING_RIGHTS:CASTLING_RIGHTS+6] = 1 
    if chessboard.has_kingside_castling_rights(chess.WHITE):
        torch_board[0, CASTLING_RIGHTS] = 0
        torch_board[0, CASTLING_RIGHTS+2] = 0
    if chessboard.has_queenside_castling_rights(chess.WHITE):
        torch_board[0, CASTLING_RIGHTS] = 0
        torch_board[0, CASTLING_RIGHTS+4] = 0
    if chessboard.has_kingside_castling_rights(chess.BLACK):
        torch_board[0, CASTLING_RIGHTS+1] = 0
        torch_board[0, CASTLING_RIGHTS+3] = 0
    if chessboard.has_queenside_castling_rights(chess.BLACK):
        torch_board[0, CASTLING_RIGHTS+1] = 0
        torch_board[0, CASTLING_RIGHTS+5] = 0

    return torch_board.to("cuda:0"), players.to("cuda:0")

