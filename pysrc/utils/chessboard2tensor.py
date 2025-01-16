import torch
import chess

def chessboard2tensor(chessboard: chess.Board):
    """ given a chess boards returns a torch tensor representation of it """

    torch_board = torch.zeros(1, 100, dtype=torch.int)
    players = torch.tensor([chessboard.turn], dtype=torch.int)
    for square, piece in chessboard.piece_map().items():
        color = piece.color if piece is not None else None
        piece_type = piece.piece_type if piece is not None else None
        match (color, piece_type):
            case (None       , None         ): torch_board[0, square] = 0
            case (chess.BLACK, chess.PAWN   ): torch_board[0, square] = 1
            case (chess.BLACK, chess.KNIGHT ): torch_board[0, square] = 2
            case (chess.BLACK, chess.BISHOP ): torch_board[0, square] = 3
            case (chess.BLACK, chess.ROOK   ): torch_board[0, square] = 4
            case (chess.BLACK, chess.QUEEN  ): torch_board[0, square] = 5
            case (chess.BLACK, chess.KING   ): torch_board[0, square] = 6
            case (chess.WHITE, chess.PAWN   ): torch_board[0, square] = 7
            case (chess.WHITE, chess.KNIGHT ): torch_board[0, square] = 8
            case (chess.WHITE, chess.BISHOP ): torch_board[0, square] = 9
            case (chess.WHITE, chess.ROOK   ): torch_board[0, square] = 10
            case (chess.WHITE, chess.QUEEN  ): torch_board[0, square] = 11
            case (chess.WHITE, chess.KING   ): torch_board[0, square] = 12
            case _ : raise ValueError(f"Invalid piece: {piece}")

    torch_board[0, 64:70] = 1 
    if chessboard.has_kingside_castling_rights(chess.WHITE):
        torch_board[0, 64] = 0
        torch_board[0, 66] = 0
    if chessboard.has_queenside_castling_rights(chess.WHITE):
        torch_board[0, 64] = 0
        torch_board[0, 68] = 0
    if chessboard.has_kingside_castling_rights(chess.BLACK):
        torch_board[0, 65] = 0
        torch_board[0, 67] = 0
    if chessboard.has_queenside_castling_rights(chess.BLACK):
        torch_board[0, 65] = 0
        torch_board[0, 69] = 0

    return torch_board, players

