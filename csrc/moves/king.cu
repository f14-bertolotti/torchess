#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool king_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs a king movement
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    // this routine does not verify if the king is in check
    
    const unsigned char player_king = boards[TURN][env] * 6 + WHITE_KING;
    const unsigned char source = actions[0][env] * 8 + actions[1][env];
    const unsigned char target = actions[2][env] * 8 + actions[3][env];
    const unsigned char enemy_pawn  = ((boards[TURN][env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((boards[TURN][env] + 1) % 2) * 6 + WHITE_QUEEN;
    const unsigned char srcrow = actions[0][env];
    const unsigned char srccol = actions[1][env];
    const unsigned char tgtrow = actions[2][env];
    const unsigned char tgtcol = actions[3][env];

    const bool is_action_ok = (
        (actions[4][env] == 0)               & // no special action
        (boards[source][env] == player_king) & // source is a king
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
            (boards[target][env] == EMPTY) |
            ((boards[target][env] >= enemy_pawn) & 
             (boards[target][env] <= enemy_queen))
        ) // target is empty or enemy
    );

    boards[target][env] = is_action_ok ? player_king : boards[target][env];
    boards[source][env] = is_action_ok ? EMPTY       : boards[source][env];
    boards[KING_POSITION + boards[TURN][env] * 2 + 0][env] = is_action_ok ? tgtrow : boards[KING_POSITION + boards[TURN][env] * 2 + 0][env];
    boards[KING_POSITION + boards[TURN][env] * 2 + 1][env] = is_action_ok ? tgtcol : boards[KING_POSITION + boards[TURN][env] * 2 + 1][env];


    return !is_action_ok;
}

__global__ void king_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = king_move(env, boards, actions);
}


