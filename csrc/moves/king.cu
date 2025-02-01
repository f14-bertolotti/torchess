#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool king_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs a king movement
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    // this routine does not verify if the king is in check
    
    const unsigned char player_king = players[env] * 6 + WHITE_KING;
    const unsigned char source = actions[env][0] * 8 + actions[env][1];
    const unsigned char target = actions[env][2] * 8 + actions[env][3];
    const unsigned char enemy_pawn  = ((players[env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((players[env] + 1) % 2) * 6 + WHITE_QUEEN;
    const unsigned char srcrow = actions[env][0];
    const unsigned char srccol = actions[env][1];
    const unsigned char tgtrow = actions[env][2];
    const unsigned char tgtcol = actions[env][3];

    const bool is_action_ok = (
        (actions[env][4] == 0)               & // no special action
        (boards[env][source] == player_king) & // source is a king
        (
            ((srcrow == tgtrow + 1) & (srccol == tgtcol + 1) & (tgtrow + 1 <= 7) & (tgtcol + 1 <= 7)) |
            ((srcrow == tgtrow + 1) & (srccol == tgtcol - 1) & (tgtrow + 1 <= 7) & (tgtcol - 1 >= 0)) |
            ((srcrow == tgtrow - 1) & (srccol == tgtcol + 1) & (tgtrow - 1 >= 0) & (tgtcol + 1 <= 7)) |
            ((srcrow == tgtrow - 1) & (srccol == tgtcol - 1) & (tgtrow - 1 >= 0) & (tgtcol - 1 >= 0)) |
            ((srcrow == tgtrow + 1) & (srccol == tgtcol    ) & (tgtrow + 1 <= 7)) |
            ((srcrow == tgtrow - 1) & (srccol == tgtcol    ) & (tgtrow - 1 >= 0)) |
            ((srccol == tgtcol + 1) & (srcrow == tgtrow    ) & (tgtcol + 1 <= 7)) |
            ((srccol == tgtcol - 1) & (srcrow == tgtrow    ) & (tgtcol - 1 >= 0))
        ) & ( // target is a valid king movement
            (boards[env][target] == EMPTY) |
            ((boards[env][target] >= enemy_pawn) & 
             (boards[env][target] <= enemy_queen))
        ) // target is empty or enemy
    );

    boards[env][target] = is_action_ok ? player_king : boards[env][target];
    boards[env][source] = is_action_ok ? EMPTY       : boards[env][source];
    boards[env][KING_POSITION + players[env] * 2 + 0] = is_action_ok ? tgtrow : boards[env][KING_POSITION + players[env] * 2 + 0];
    boards[env][KING_POSITION + players[env] * 2 + 1] = is_action_ok ? tgtcol : boards[env][KING_POSITION + players[env] * 2 + 1];


    return !is_action_ok;
}

__global__ void king_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = king_move(env, players, boards, actions);
}


