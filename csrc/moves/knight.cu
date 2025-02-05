#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool knight_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs a knight movement
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    
    const unsigned char player_knight = players[env] * 6 + WHITE_KNIGHT;
    const unsigned char source = actions[0][env] * 8 + actions[1][env];
    const unsigned char target = actions[2][env] * 8 + actions[3][env];
    const unsigned char enemy_pawn  = ((players[env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((players[env] + 1) % 2) * 6 + WHITE_QUEEN;
    const unsigned char srcrow = actions[0][env];
    const unsigned char srccol = actions[1][env];
    const unsigned char tgtrow = actions[2][env];
    const unsigned char tgtcol = actions[3][env];

    const bool is_action_ok = (
        (actions[4][env] == 0)                 & // no special action
        (boards[source][env] == player_knight) & // source is a knight
        (
            ((srcrow == tgtrow + 2) & (srccol == tgtcol + 1) & (tgtrow + 2 <= 7) & (tgtcol + 1 <= 7)) |
            ((srcrow == tgtrow + 2) & (srccol == tgtcol - 1) & (tgtrow + 2 <= 7) & (tgtcol - 1 >= 0)) |
            ((srcrow == tgtrow - 2) & (srccol == tgtcol + 1) & (tgtrow - 2 >= 0) & (tgtcol + 1 <= 7)) |
            ((srcrow == tgtrow - 2) & (srccol == tgtcol - 1) & (tgtrow - 2 >= 0) & (tgtcol - 1 >= 0)) |
            ((srcrow == tgtrow + 1) & (srccol == tgtcol + 2) & (tgtrow + 1 <= 7) & (tgtcol + 2 <= 7)) |
            ((srcrow == tgtrow + 1) & (srccol == tgtcol - 2) & (tgtrow + 1 <= 7) & (tgtcol - 2 >= 0)) |
            ((srcrow == tgtrow - 1) & (srccol == tgtcol + 2) & (tgtrow - 1 >= 0) & (tgtcol + 2 <= 7)) |
            ((srcrow == tgtrow - 1) & (srccol == tgtcol - 2) & (tgtrow - 1 >= 0) & (tgtcol - 2 >= 0))
        ) & ( // target is a valid knight movement
            (boards[target][env] == EMPTY) |
            ((boards[target][env] >= enemy_pawn) & 
             (boards[target][env] <= enemy_queen))
        ) // target is empty or enemy
    );

    boards[target][env] = is_action_ok ? player_knight : boards[target][env];
    boards[source][env] = is_action_ok ? EMPTY         : boards[source][env];

    return !is_action_ok;
}

__global__ void knight_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = knight_move(env, players, boards, actions);
}


