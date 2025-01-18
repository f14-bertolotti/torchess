#pragma once
#include <torch/extension.h>
#include "../chess-consts.h"

__device__ unsigned char pawn_movement(
    int env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs pawn promotion
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    
    const unsigned char player_pawn = players[env] * 6 + WHITE_PAWN;
    const unsigned char source = actions[env][0] * 8 + actions[env][1];
    const unsigned char target = actions[env][2] * 8 + actions[env][3];

    const bool is_action_ok = (
        (actions[env][4] == 0              ) & // no special action
        (boards[env][source] == player_pawn) & // moving a pawn
        (boards[env][target] == EMPTY      ) & // target is empty
        (target >= 8                       ) & // not in first row (would be a promotion)
        (target <= 55                      ) & // not in last  row (would be a promotion)
        ((
            ( players[env] == WHITE ) & // if moving a white
            ( source == target + 8  )   // moving forward
        ) | (
            ( players[env] == BLACK ) & // if moving a black
            ( source == target - 8  )   // moving forward
        ))
    );

    boards[env][target] = is_action_ok ? player_pawn : boards[env][target];
    boards[env][source] = is_action_ok ? EMPTY       : boards[env][source];

    return !is_action_ok;
}

__global__ void pawn_move_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = pawn_movement(env, players, boards, actions);
}


