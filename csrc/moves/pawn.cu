#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool pawn_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs pawn promotion
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    
    const unsigned char player_pawn = boards[TURN][env] * 6 + WHITE_PAWN;
    const unsigned char source = actions[0][env] * 8 + actions[1][env];
    const unsigned char target = actions[2][env] * 8 + actions[3][env];
    const unsigned char enemy_pawn  = ((boards[TURN][env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((boards[TURN][env] + 1) % 2) * 6 + WHITE_QUEEN;
    const   signed char ahead = boards[TURN][env] == WHITE ? -1 : 1;

    const bool is_action_ok = (
        (actions[4][env] == 0              ) & // no special action
        (boards[source][env] == player_pawn) & // moving a pawn
        (target >= 8                       ) & // not in first row (would be a promotion)
        (target <= 55                      ) & // not in last  row (would be a promotion)
        (actions[2][env]-actions[0][env] == ahead) & // one step ahead
        ((
            (actions[1][env] == actions[3][env]        ) & // pawn moving forward
            (boards[target][env] == EMPTY              )   // action target is empty
        ) | (
            (actions[1][env] == actions[3][env] - 1) & // pawn capturing left
            (boards[target][env] >= enemy_pawn     ) & // action target is not empty
            (boards[target][env] <= enemy_queen    )   // action target is an enemy piece
        ) | (
            (actions[1][env] == actions[3][env] + 1) & // pawn capturing right
            (boards[target][env] >= enemy_pawn     ) & // action target is not empty
            (boards[target][env] <= enemy_queen    )   // action target is an enemy piece
        ))
    );

    boards[target][env] = is_action_ok ? player_pawn : boards[target][env];
    boards[source][env] = is_action_ok ? EMPTY       : boards[source][env];

    return !is_action_ok;
}

__global__ void pawn_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = pawn_move(env, boards, actions);
}


