#pragma once
#include <torch/extension.h>
#include "../chess_attacks.cu"
#include "../chess_consts.h"

__device__ bool kingside_castle_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs kingside castling action
    // returns 0 if everything is ok
    // returns 1 if the action was a kingside castling but the conditions were not met
    
    const unsigned char player_king = boards[TURN][env] * 6 + WHITE_KING;
    const unsigned char player_rook = boards[TURN][env] * 6 + WHITE_ROOK;
    const unsigned char special = actions[4][env];
    const unsigned char castle_row  = boards[TURN][env] == WHITE ? 7 : 0;
    const unsigned char king_source = castle_row * 8 + 4;
    const unsigned char rook_source = castle_row * 8 + 7;
    const unsigned char king_target = castle_row * 8 + 6;
    const unsigned char rook_target = castle_row * 8 + 5;

    const bool is_kingside_castle = (
        (actions[0][env] == 0   ) & // action source empty
        (actions[1][env] == 0   ) & // action source empty
        (actions[2][env] == 0   ) & // action target empty
        (actions[3][env] == 0   ) & // action target empty
        (special == KING_CASTLE )   // king castling action
    );

    const bool is_action_ok = ( 
        is_kingside_castle                                        & // kingside castling action
        (boards[KING_MOVED + boards[TURN][env]][env] == 0            ) & // king has not moved
        (boards[KINGSIDE_ROOK_MOVED + boards[TURN][env]][env] == 0   ) & // king-side rook has not moved
        (boards[king_source][env] == player_king                ) & // king is in the right position
        (boards[rook_target][env] == EMPTY                      ) & // king-side is empty
        (boards[king_target][env] == EMPTY                      ) & // king-side is empty
        (boards[rook_source][env] == player_rook                ) & // king-side rook is in the right position
        (count_attacks(env, castle_row, 4, boards) == 0) & // king is not in check
        (count_attacks(env, castle_row, 5, boards) == 0) & // king-side 1 is not in check
        (count_attacks(env, castle_row, 6, boards) == 0)   // king-side 2 is not in check
    );

    boards[king_source][env] = is_action_ok ? EMPTY       : boards[king_source][env];
    boards[rook_source][env] = is_action_ok ? EMPTY       : boards[rook_source][env];
    boards[rook_target][env] = is_action_ok ? player_rook : boards[rook_target][env];
    boards[king_target][env] = is_action_ok ? player_king : boards[king_target][env];

    // update king stored position
    boards[KING_POSITION + boards[TURN][env] * 2 + 0][env] = is_action_ok ? castle_row : boards[KING_POSITION + boards[TURN][env] * 2 + 0][env];
    boards[KING_POSITION + boards[TURN][env] * 2 + 1][env] = is_action_ok ? 6          : boards[KING_POSITION + boards[TURN][env] * 2 + 1][env];

    return !is_action_ok;
}

__global__ void kingside_castle_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = kingside_castle_move(env, boards, actions);
}
