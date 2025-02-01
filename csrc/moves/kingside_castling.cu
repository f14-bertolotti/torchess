#pragma once
#include <torch/extension.h>
#include "../chess_attacks.cu"
#include "../chess_consts.h"

__device__ bool kingside_castle_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs kingside castling action
    // returns 0 if everything is ok
    // returns 1 if the action was a kingside castling but the conditions were not met
    
    const unsigned char player_king = players[env] * 6 + WHITE_KING;
    const unsigned char player_rook = players[env] * 6 + WHITE_ROOK;
    const unsigned char special = actions[env][4];
    const unsigned char castle_row  = players[env] == WHITE ? 7 : 0;
    const unsigned char king_source = castle_row * 8 + 4;
    const unsigned char rook_source = castle_row * 8 + 7;
    const unsigned char king_target = castle_row * 8 + 6;
    const unsigned char rook_target = castle_row * 8 + 5;

    const bool is_kingside_castle = (
        (actions[env][0] == 0   ) & // action source empty
        (actions[env][1] == 0   ) & // action source empty
        (actions[env][2] == 0   ) & // action target empty
        (actions[env][3] == 0   ) & // action target empty
        (special == KING_CASTLE )   // king castling action
    );

    const bool is_action_ok = ( 
        is_kingside_castle                                        & // kingside castling action
        (boards[env][KING_MOVED + players[env]] == 0            ) & // king has not moved
        (boards[env][KINGSIDE_ROOK_MOVED + players[env]] == 0   ) & // king-side rook has not moved
        (boards[env][king_source] == player_king                ) & // king is in the right position
        (boards[env][rook_target] == EMPTY                      ) & // king-side is empty
        (boards[env][king_target] == EMPTY                      ) & // king-side is empty
        (boards[env][rook_source] == player_rook                ) & // king-side rook is in the right position
        (count_attacks(env, castle_row, 4, players, boards) == 0) & // king is not in check
        (count_attacks(env, castle_row, 5, players, boards) == 0) & // king-side 1 is not in check
        (count_attacks(env, castle_row, 6, players, boards) == 0)   // king-side 2 is not in check
    );

    boards[env][king_source] = is_action_ok ? EMPTY       : boards[env][king_source];
    boards[env][rook_source] = is_action_ok ? EMPTY       : boards[env][rook_source];
    boards[env][rook_target] = is_action_ok ? player_rook : boards[env][rook_target];
    boards[env][king_target] = is_action_ok ? player_king : boards[env][king_target];

    // update king stored position
    boards[env][KING_POSITION + players[env] * 2 + 0] = is_action_ok ? castle_row : boards[env][KING_POSITION + players[env] * 2 + 0];
    boards[env][KING_POSITION + players[env] * 2 + 1] = is_action_ok ? 6          : boards[env][KING_POSITION + players[env] * 2 + 1];

    return !is_action_ok;
}

__global__ void kingside_castle_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = kingside_castle_move(env, players, boards, actions);
}
