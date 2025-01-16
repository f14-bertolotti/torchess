#pragma once

// board pieces
#define EMPTY        0 
#define WHITE_PAWN   1 
#define WHITE_KNIGHT 2 
#define WHITE_BISHOP 3 
#define WHITE_ROOK   4 
#define WHITE_QUEEN  5 
#define WHITE_KING   6 
#define BLACK_PAWN   7 
#define BLACK_KNIGHT 8 
#define BLACK_BISHOP 9 
#define BLACK_ROOK   10
#define BLACK_QUEEN  11
#define BLACK_KING   12

// turn colors
#define WHITE 0
#define BLACK 1

// action types
#define MOVE             0
#define KING_CASTLE      1
#define QUEEN_CASTLE     2
#define EN_PASSANT       3
#define PROMOTION_QUEEN  4
#define PROMOTION_ROOK   5
#define PROMOTION_BISHOP 6
#define PROMOTION_KNIGHT 7

// special memory location
#define KING_MOVED 64 // has the king moved 
#define KINGSIDE_ROOK_MOVED 66 // has the kingside rook moved
#define QUEENSIDE_ROOK_MOVED 68 // has the queenside rook moved
#define PREV_ACTION 70 // previous action
