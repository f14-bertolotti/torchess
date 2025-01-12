import torch
import chess
import pawner

chess_board = chess.Board()

def to_torch_tensor(chess_board: chess.Board) -> torch.Tensor:
    torch_board = torch.zeros(1, 66, dtype=torch.long)
    for square, piece in chess_board.piece_map().items():
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
    return torch_board

def attacks_board(chess_board: chess.Board, color: chess.WHITE | chess.BLACK) -> torch.Tensor:
    attacks = torch.zeros(1, 64, dtype=torch.int64)
    for square in chess.SQUARES:
        attacks[0, square] += len(chess_board.attackers(color, square))
    return attacks

print("pawner")
attacks_tensor = torch.zeros(1, 64, dtype=torch.long).to("cuda:0")
player_tensor = torch.zeros(1, dtype=torch.long).to("cuda:0")
board_tensor = to_torch_tensor(chess_board).to("cuda:0")

print(board_tensor[:,:-2].view(8,8))
pawner.attacks(
    board_tensor, 
    player_tensor, 
    attacks_tensor
)
print(attacks_tensor.view(8,8))

print("python-chess")
print(attacks_board(chess_board, chess.WHITE).view(8,8))

