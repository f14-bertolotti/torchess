#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool promotion_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs pawn promotion
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    
    const unsigned char promotion_row = boards[TURN][env] == WHITE ? 0 : 7;
    const unsigned char starting_row  = boards[TURN][env] == WHITE ? 1 : 6;
    const unsigned char player_pawn = boards[TURN][env] * 6 + WHITE_PAWN;
    const unsigned char source = actions[0][env] * 8 + actions[1][env];
    const unsigned char target = actions[2][env] * 8 + actions[3][env];
    const unsigned char enemy_pawn  = ((boards[TURN][env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((boards[TURN][env] + 1) % 2) * 6 + WHITE_QUEEN;

    const bool is_action_ok = (
        (actions[4][env] >= PROMOTION_QUEEN & actions[4][env] <= PROMOTION_KNIGHT) & // action is a pawn promotion
        (actions[0][env] == starting_row     ) & // action source is in pre-promotion row
        (actions[2][env] == promotion_row    ) & // action target is in promotion row
        (boards[source][env] == player_pawn  ) & // action source is a pawn
        ((
            actions[1][env] == actions[3][env] & // pawn moving forward
            boards[target][env] == EMPTY         // action target is empty
        ) | (
            actions[1][env] == actions[3][env] - 1 & // pawn capturing left
            boards[target][env] >= enemy_pawn      & // action target is not empty
            boards[target][env] <= enemy_queen       // action target is an enemy piece
        ) | (
            actions[1][env] == actions[3][env] + 1 & // pawn capturing right
            boards[target][env] >= enemy_pawn      & // action target is not empty
            boards[target][env] <= enemy_queen       // action target is an enemy piece
        ))
    );

    boards[target][env] = (is_action_ok & (actions[4][env] == PROMOTION_QUEEN )) ? boards[TURN][env] * 6 + WHITE_QUEEN  : boards[target][env];
    boards[target][env] = (is_action_ok & (actions[4][env] == PROMOTION_KNIGHT)) ? boards[TURN][env] * 6 + WHITE_KNIGHT : boards[target][env];
    boards[target][env] = (is_action_ok & (actions[4][env] == PROMOTION_ROOK  )) ? boards[TURN][env] * 6 + WHITE_ROOK   : boards[target][env];
    boards[target][env] = (is_action_ok & (actions[4][env] == PROMOTION_BISHOP)) ? boards[TURN][env] * 6 + WHITE_BISHOP : boards[target][env];
    boards[source][env] = (is_action_ok) ? EMPTY : boards[source][env];
    
    return !is_action_ok;
}

__global__ void promotion_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = promotion_move(env, boards, actions);
}


