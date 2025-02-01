#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"
#include "../clamp.cu"

__device__ bool doublepush_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs pawn double move
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    
    const unsigned char player_pawn = players[env] * 6 + WHITE_PAWN;
    const unsigned char source = actions[env][0] * 8 + actions[env][1];
    const unsigned char target = actions[env][2] * 8 + actions[env][3];
    const unsigned char player_1st_row = players[env] == WHITE ? 6 : 1;
    const unsigned char player_3rd_row = players[env] == WHITE ? 4 : 3;

    const bool is_action_ok = (
        (actions[env][4] == 0               ) & // no special move
        (boards[env][source] == player_pawn ) & // moving a pawn
        (actions[env][0] == player_1st_row  ) & // from the first row
        (actions[env][2] == player_3rd_row  ) & // to the third row
        (actions[env][1] == actions[env][3] ) & // moving in the same column
        (boards[env][target] == EMPTY       ) & // action target is empty
        (boards[env][clamp(0,63,source + ((+8) * players[env] + (-8) * (1-players[env])))] == EMPTY) // intermediate cell is empty
    );

    boards[env][target] = is_action_ok ? player_pawn : boards[env][target];
    boards[env][source] = is_action_ok ? EMPTY       : boards[env][source];
    
    return !is_action_ok;
}

__global__ void doublepush_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = doublepush_move(env, players, boards, actions);
}


