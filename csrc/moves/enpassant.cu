#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"

__device__ bool enpassant_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs en passant
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    
    const unsigned char player_pawn = players[env] * 6 + WHITE_PAWN;
    const unsigned char enemy_pawn  = ((players[env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char source = actions[0][env] * 8 + actions[1][env];
    const unsigned char target = actions[2][env] * 8 + actions[3][env];
    const unsigned char prev_action = WHITE_PREV1 + 10*((players[env]+1)%2);
    const unsigned char prev_target = boards[prev_action+2][env] * 8 + boards[prev_action+3][env];
    const unsigned char enpassant_src_row = players[env] == WHITE ? 3 : 4;
    const unsigned char enpassant_tgt_row = players[env] == WHITE ? 2 : 5;

    const bool is_action_ok = (
        (actions[4][env] == 0                                            ) & // no special action
        (actions[0][env] == enpassant_src_row                            ) & // action source is in en passant row
        (actions[2][env] == enpassant_tgt_row                            ) & // action target is in en passant row
        (abs(actions[1][env] - actions[3][env]) == 1                     ) & // moving on side column
        (boards[source][env] == player_pawn                              ) & // moving a pawn
        (boards[prev_action+4][env] == 0                                 ) & // previous action was a normal action 
        (boards[prev_action+3][env] == actions[3][env]                   ) & // previous action was a to the same column
        (abs(boards[prev_action][env] - boards[prev_action+2][env]) == 2 ) & // previous action was a double move
        (boards[prev_target][env] == enemy_pawn                          ) & // previous action moved a pawn
        (boards[target][env] == EMPTY                                    )   // action target is empty
    );

    boards[target][env] = is_action_ok ? player_pawn : boards[target][env];
    boards[source][env] = is_action_ok ? EMPTY       : boards[source][env];
    boards[prev_target][env] = is_action_ok ? EMPTY : boards[prev_target][env];

    return !is_action_ok;
}

__global__ void enpassant_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = enpassant_move(env, players, boards, actions);
}


