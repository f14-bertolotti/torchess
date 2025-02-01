#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool promotion_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs pawn promotion
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    
    const unsigned char promotion_row = players[env] == WHITE ? 0 : 7;
    const unsigned char starting_row  = players[env] == WHITE ? 1 : 6;
    const unsigned char player_pawn = players[env] * 6 + WHITE_PAWN;
    const unsigned char source = actions[env][0] * 8 + actions[env][1];
    const unsigned char target = actions[env][2] * 8 + actions[env][3];
    const unsigned char enemy_pawn  = ((players[env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((players[env] + 1) % 2) * 6 + WHITE_QUEEN;

    const bool is_action_ok = (
        (actions[env][4] >= PROMOTION_QUEEN & actions[env][4] <= PROMOTION_KNIGHT) & // action is a pawn promotion
        (actions[env][0] == starting_row     ) & // action source is in pre-promotion row
        (actions[env][2] == promotion_row    ) & // action target is in promotion row
        (boards[env][source] == player_pawn  ) & // action source is a pawn
        ((
            actions[env][1] == actions[env][3] & // pawn moving forward
            boards[env][target] == EMPTY         // action target is empty
        ) | (
            actions[env][1] == actions[env][3] - 1 & // pawn capturing left
            boards[env][target] >= enemy_pawn      & // action target is not empty
            boards[env][target] <= enemy_queen       // action target is an enemy piece
        ) | (
            actions[env][1] == actions[env][3] + 1 & // pawn capturing right
            boards[env][target] >= enemy_pawn      & // action target is not empty
            boards[env][target] <= enemy_queen       // action target is an enemy piece
        ))
    );

    boards[env][target] = (is_action_ok & (actions[env][4] == PROMOTION_QUEEN )) ? players[env] * 6 + WHITE_QUEEN  : boards[env][target];
    boards[env][target] = (is_action_ok & (actions[env][4] == PROMOTION_KNIGHT)) ? players[env] * 6 + WHITE_KNIGHT : boards[env][target];
    boards[env][target] = (is_action_ok & (actions[env][4] == PROMOTION_ROOK  )) ? players[env] * 6 + WHITE_ROOK   : boards[env][target];
    boards[env][target] = (is_action_ok & (actions[env][4] == PROMOTION_BISHOP)) ? players[env] * 6 + WHITE_BISHOP : boards[env][target];
    boards[env][source] = (is_action_ok) ? EMPTY : boards[env][source];
    
    return !is_action_ok;
}

__global__ void promotion_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = promotion_move(env, players, boards, actions);
}


