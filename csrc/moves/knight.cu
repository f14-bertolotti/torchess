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
    const unsigned char source = actions[env][0] * 8 + actions[env][1];
    const unsigned char target = actions[env][2] * 8 + actions[env][3];
    const unsigned char enemy_pawn  = ((players[env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((players[env] + 1) % 2) * 6 + WHITE_QUEEN;
    const unsigned char srcrow = actions[env][0];
    const unsigned char srccol = actions[env][1];
    const unsigned char tgtrow = actions[env][2];
    const unsigned char tgtcol = actions[env][3];

    const bool is_action_ok = (
        (actions[env][4] == 0)                 & // no special action
        (boards[env][source] == player_knight) & // source is a knight
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
            (boards[env][target] == EMPTY) |
            ((boards[env][target] >= enemy_pawn) & 
             (boards[env][target] <= enemy_queen))
        ) // target is empty or enemy
    );

    boards[env][target] = is_action_ok ? player_knight : boards[env][target];
    boards[env][source] = is_action_ok ? EMPTY         : boards[env][source];

    return !is_action_ok;
}

__global__ void knight_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = knight_move(env, players, boards, actions);
}


