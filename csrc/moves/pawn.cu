#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool pawn_move(
    size_t env,
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
    const unsigned char enemy_pawn  = ((players[env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((players[env] + 1) % 2) * 6 + WHITE_QUEEN;
    const   signed char ahead = players[env] == WHITE ? -1 : 1;

    const bool is_action_ok = (
        (actions[env][4] == 0              ) & // no special action
        (boards[env][source] == player_pawn) & // moving a pawn
        (target >= 8                       ) & // not in first row (would be a promotion)
        (target <= 55                      ) & // not in last  row (would be a promotion)
        (actions[env][2]-actions[env][0] == ahead) & // one step ahead
        ((
            (actions[env][1] == actions[env][3]        ) & // pawn moving forward
            (boards[env][target] == EMPTY              )   // action target is empty
        ) | (
            (actions[env][1] == actions[env][3] - 1) & // pawn capturing left
            (boards[env][target] >= enemy_pawn     ) & // action target is not empty
            (boards[env][target] <= enemy_queen    )   // action target is an enemy piece
        ) | (
            (actions[env][1] == actions[env][3] + 1) & // pawn capturing right
            (boards[env][target] >= enemy_pawn     ) & // action target is not empty
            (boards[env][target] <= enemy_queen    )   // action target is an enemy piece
        ))
    );

    boards[env][target] = is_action_ok ? player_pawn : boards[env][target];
    boards[env][source] = is_action_ok ? EMPTY       : boards[env][source];

    return !is_action_ok;
}

__global__ void pawn_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = pawn_move(env, players, boards, actions);
}


