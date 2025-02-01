import chess
import torch

def san2pwn(action,board,turn)->torch.Tensor:
    """Convert a SAN action to a pawner action."""
    src, dst = action[:2], action[2:]
    src_row, src_col = 8-int(src[1]), ord(src[0])-97
    dst_row, dst_col = 8-int(dst[1]), ord(dst[0])-97
    turn = 0 if turn == chess.WHITE else 1

    if len(action) == 5:
        if action[4] == "q": return torch.tensor([src_row, src_col, dst_row, dst_col, 3], dtype=torch.int, device="cuda:0") # promotion queen
        if action[4] == "r": return torch.tensor([src_row, src_col, dst_row, dst_col, 4], dtype=torch.int, device="cuda:0") # promotion rook
        if action[4] == "b": return torch.tensor([src_row, src_col, dst_row, dst_col, 5], dtype=torch.int, device="cuda:0") # promotion bishop
        if action[4] == "n": return torch.tensor([src_row, src_col, dst_row, dst_col, 6], dtype=torch.int, device="cuda:0") # promotion knight

    if board.piece_at(board.parse_san(action).from_square).piece_type == chess.KING:
        if action == "e1g1": return torch.tensor([0, 0, 0, 0, 1], dtype=torch.int, device="cuda:0") # kingside castle white
        if action == "e8g8": return torch.tensor([0, 0, 0, 0, 1], dtype=torch.int, device="cuda:0") # kingside castle black
        if action == "e1c1": return torch.tensor([0, 0, 0, 0, 2], dtype=torch.int, device="cuda:0") # queenside castle white
        if action == "e8c8": return torch.tensor([0, 0, 0, 0, 2], dtype=torch.int, device="cuda:0") # queenside castle black

    return torch.tensor([src_row, src_col, dst_row, dst_col, 0], dtype=torch.int, device="cuda:0")
