#pragma once
#include <torch/extension.h>
#include "chess-consts.h"
#include "moves/kingside-castling.cu"
#include "moves/queenside-castling.cu"
#include "moves/promotion.cu"
#include "moves/pawn-move.cu"
#include "moves/double-move.cu"
#include "moves/en-passant.cu"
#include "moves/knight-move.cu"
#include "moves/king-move.cu"
#include "moves/rook-move.cu"
#include "moves/bishop-move.cu"
#include "moves/queen-move.cu"
#include "chess-attacks.cu"

__global__ void step_kernel(
    torch::PackedTensorAccessor32<int  , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int  , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int  , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<float, 1 , torch::RestrictPtrTraits> rewards ,
    torch::PackedTensorAccessor32<bool , 1 , torch::RestrictPtrTraits> dones
) {
    // performs a standard action
    // returns 0 if everything is ok
    // returns 1 if the action was a standard action but the conditions were not met
    const int env = blockIdx.x * blockDim.x + threadIdx.x;

    const bool is_action_ok = !(
        kingside_castle (env, players, boards, actions) &
        queenside_castle(env, players, boards, actions) &
        pawn_promotion  (env, players, boards, actions) &
        pawn_movement   (env, players, boards, actions) &
        doublemove      (env, players, boards, actions) &
        en_passant      (env, players, boards, actions) &
        knight_movement (env, players, boards, actions) &
        king_movement   (env, players, boards, actions) &
        rook_movement   (env, players, boards, actions) &
        bishop_movement (env, players, boards, actions) &
        queen_movement  (env, players, boards, actions)
    );

    // current player king not attacked
    const unsigned char king_row = boards[env][KING_POSITION + players[env] * 2 + 0];
    const unsigned char king_col = boards[env][KING_POSITION + players[env] * 2 + 1];
    const bool is_king_ok = !(count_attacks(env, king_row, king_col, players, boards) > 0);

    
    // check if the game is over
    const unsigned char enemy_king_row = boards[env][KING_POSITION + (players[env] + 1) % 2 * 2 + 0];
    const unsigned char enemy_king_col = boards[env][KING_POSITION + (players[env] + 1) % 2 * 2 + 1];
    const bool is_winning_action = (
        (
            (enemy_king_row < 7) & ((boards[env][clamp(0,63,(enemy_king_row+1) * 8 + enemy_king_col)] != EMPTY) | (count_attacks(env, enemy_king_row+1, enemy_king_col, players, boards) > 0))
        ) & ( // down movement not possible
            (enemy_king_row > 1) & ((boards[env][clamp(0,63,(enemy_king_row-1) * 8 + enemy_king_col)] != EMPTY) | (count_attacks(env, enemy_king_row-1, enemy_king_col, players, boards) > 0))
        ) & ( // up movement not possible
            (enemy_king_col < 7) & ((boards[env][clamp(0,63,enemy_king_row * 8 + enemy_king_col + 1)] != EMPTY) | (count_attacks(env, enemy_king_row, enemy_king_col+1, players, boards) > 0))
        ) & ( // right movement not possible
            (enemy_king_col > 1) & ((boards[env][clamp(0,63,enemy_king_row * 8 + enemy_king_col - 1)] != EMPTY) | (count_attacks(env, enemy_king_row, enemy_king_col-1, players, boards) > 0))
        ) & (// left movement not possible
            (enemy_king_row < 7) & (enemy_king_col < 7) & ((boards[env][clamp(0,63,(enemy_king_row+1) * 8 + enemy_king_col + 1)] != EMPTY) | (count_attacks(env, enemy_king_row+1, enemy_king_col+1, players, boards) > 0))
        ) & (// down-right movement not possible
            (enemy_king_row > 1) & (enemy_king_col > 1) & ((boards[env][clamp(0,63,(enemy_king_row-1) * 8 + enemy_king_col - 1)] != EMPTY) | (count_attacks(env, enemy_king_row-1, enemy_king_col-1, players, boards) > 0))
        ) & (// up-left movement not possible
            (enemy_king_row < 7) & (enemy_king_col < 7) & ((boards[env][clamp(0,63,(enemy_king_row+1) * 8 + enemy_king_col + 1)] != EMPTY) | (count_attacks(env, enemy_king_row+1, enemy_king_col+1, players, boards) > 0))
        ) & (// down-left movement not possible
            (enemy_king_row > 1) & (enemy_king_col < 7) & ((boards[env][clamp(0,63,(enemy_king_row-1) * 8 + enemy_king_col + 1)] != EMPTY) | (count_attacks(env, enemy_king_row-1, enemy_king_col+1, players, boards) > 0))
        ) & (// up-right movement not possible
            (enemy_king_row < 7) & (enemy_king_col > 1) & ((boards[env][clamp(0,63,(enemy_king_row+1) * 8 + enemy_king_col - 1)] != EMPTY) | (count_attacks(env, enemy_king_row+1, enemy_king_col-1, players, boards) > 0))
        )
    );

    rewards[env] = (is_action_ok & is_king_ok &  is_winning_action) + 
                   (is_action_ok & is_king_ok & !is_winning_action) * 0 + 
                   (!is_action_ok | !is_king_ok) * -1;
    
    dones[env] = is_winning_action | !is_action_ok | !is_king_ok;

}



    

