import chess

def pwn2san(action,turn):
    """Convert a pawner action to a FEN action."""
    src_row, src_col, dst_row, dst_col, spc = action
    src_row, dst_row = 7 - src_row, 7 - dst_row
    src = "".join((chr(src_col+97), str(src_row+1)))
    dst = "".join((chr(dst_col+97), str(dst_row+1)))
    turn = 0 if turn == chess.WHITE else 1
    match spc,turn,src_row,src_col,dst_row,dst_col:
        case 0,_,_,_,_,_: return f"{src}{dst}"  # normal action
        case 1,0,0,0,0,0: return "e1g1"         # kingside castle white
        case 1,1,0,0,0,0: return "e8g8"         # kingside castle black
        case 2,0,0,0,0,0: return "e1c1"         # queenside castle white
        case 2,1,0,0,0,0: return "e8c8"         # queenside castle black
        case 3,_,_,_,_,_: return f"{src}{dst}q" # promotion queen
        case 4,_,_,_,_,_: return f"{src}{dst}r" # promotion rook
        case 5,_,_,_,_,_: return f"{src}{dst}b" # promotion bishop
        case 6,_,_,_,_,_: return f"{src}{dst}n" # promotion knight
        case _,_,_,_,_,_: raise ValueError(f"Invalid special action {spc}")


