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
    const unsigned char source = actions[0][env] * 8 + actions[1][env];
    const unsigned char target = actions[2][env] * 8 + actions[3][env];
    const unsigned char player_1st_row = players[env] == WHITE ? 6 : 1;
    const unsigned char player_3rd_row = players[env] == WHITE ? 4 : 3;

    const bool is_action_ok = (
        (actions[4][env] == 0               ) & // no special move
        (boards[source][env] == player_pawn ) & // moving a pawn
        (actions[0][env] == player_1st_row  ) & // from the first row
        (actions[2][env] == player_3rd_row  ) & // to the third row
        (actions[1][env] == actions[3][env] ) & // moving in the same column
        (boards[target][env] == EMPTY       ) & // action target is empty
        (boards[clamp(0,63,source + ((+8) * players[env] + (-8) * (1-players[env])))][env] == EMPTY) // intermediate cell is empty
    );

    boards[target][env] = is_action_ok ? player_pawn : boards[target][env];
    boards[source][env] = is_action_ok ? EMPTY       : boards[source][env];
    
    return !is_action_ok;
}

__global__ void doublepush_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = doublepush_move(env, players, boards, actions);
}


