#pragma once
#include <torch/extension.h>
#include "../chess_consts.h"
#include "../clamp.cu"

__device__ bool queen_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs a queen movement
    // returns 0 if the action was performed
    // returns 1 if the action was not applicable
    // this routine does not verify if the queen is in check
    
    const unsigned char player_queen = players[env] * 6 + WHITE_QUEEN;
    const unsigned char source = actions[env][0] * 8 + actions[env][1];
    const unsigned char target = actions[env][2] * 8 + actions[env][3];
    const unsigned char enemy_pawn  = ((players[env] + 1) % 2) * 6 + WHITE_PAWN;
    const unsigned char enemy_queen = ((players[env] + 1) % 2) * 6 + WHITE_QUEEN;
    const unsigned char srcrow = actions[env][0];
    const unsigned char srccol = actions[env][1];
    const unsigned char tgtrow = actions[env][2];
    const unsigned char tgtcol = actions[env][3];

    const char dir_x = (+1) * (srccol < tgtcol) + (-1) * (srccol > tgtcol);
    const char dir_y = (+1) * (srcrow < tgtrow) + (-1) * (srcrow > tgtrow);
    bool is_jumping_over = false;
    bool encountered_target = false;
    for (int i = 1; i < 8; i++) {
        encountered_target = encountered_target | ((srcrow + i * dir_y == tgtrow) & (srccol + i * dir_x == tgtcol));
        is_jumping_over = is_jumping_over | ((!encountered_target) & (boards[env][clamp(0,63,(srcrow + i * dir_y) * 8 + (srccol + i * dir_x))] != EMPTY));
    }

    const bool is_action_ok = (
        (actions[env][4] == 0)                & // no special action
        (boards[env][source] == player_queen) & // source is a queen
        !is_jumping_over & (                    // queen is not jumping over other pieces
            ((srcrow == tgtrow) & (srccol <= 7)) |
            ((srccol == tgtcol) & (srcrow <= 7)) |
            (abs(srcrow - tgtrow) == abs(srccol - tgtcol))
        ) & ( // target is a valid queen movement
            (boards[env][target] == EMPTY) |
            ((boards[env][target] >= enemy_pawn) & (boards[env][target] <= enemy_queen))
        ) // target is empty or enemy
    );

    boards[env][target] = is_action_ok ? player_queen : boards[env][target];
    boards[env][source] = is_action_ok ? EMPTY       : boards[env][source];

    return !is_action_ok;
}

__global__ void queen_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(0)) result[env] = queen_move(env, players, boards, actions);
}


