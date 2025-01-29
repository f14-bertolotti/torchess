#pragma once
#include <torch/extension.h>
#include "chess-consts.h"
#include "moves/kingside-castling.cu"
#include "moves/queenside-castling.cu"
#include "moves/promotion.cu"
#include "moves/pawn.cu"
#include "moves/doublepush.cu"
#include "moves/enpassant.cu"
#include "moves/knight.cu"
#include "moves/king.cu"
#include "moves/rook.cu"
#include "moves/bishop.cu"
#include "moves/queen.cu"
#include "chess-attacks.cu"

__global__ void step_kernel(
    torch::PackedTensorAccessor32<int  , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int  , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int  , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<float, 2 , torch::RestrictPtrTraits> rewards ,
    torch::PackedTensorAccessor32<bool , 1 , torch::RestrictPtrTraits> dones
) {
    // performs a standard action
    // returns 0 if everything is ok
    // returns 1 if the action was a standard action but the conditions were not met
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned char source = actions[env][0] * 8 + actions[env][1];
    const unsigned char target = actions[env][2] * 8 + actions[env][3];
    if (env < boards.size(0)) {

        // make action
        const bool is_action_ok = 
            (((actions[env][4] != 0) | (source != target))   &
            !(
                kingside_castle_move  ( env, players, boards, actions) &
                queenside_castle_move ( env, players, boards, actions) &
                promotion_move        ( env, players, boards, actions) &
                pawn_move             ( env, players, boards, actions) &
                doublepush_move       ( env, players, boards, actions) &
                enpassant_move        ( env, players, boards, actions) &
                knight_move           ( env, players, boards, actions) &
                king_move             ( env, players, boards, actions) &
                rook_move             ( env, players, boards, actions) &
                bishop_move           ( env, players, boards, actions) &
                queen_move            ( env, players, boards, actions)
             )
        );
        
        // current player king not attacked
        const unsigned char king_row = boards[env][KING_POSITION + players[env] * 2 + 0];
        const unsigned char king_col = boards[env][KING_POSITION + players[env] * 2 + 1];
        const bool is_king_ok = count_attacks(env, king_row, king_col, players, boards) == 0;
        const unsigned char player = players[env];
        const unsigned char enemy  = (players[env] + 1) % 2;

         // zero reward if action ok
         // the action was not allowed or uncovered the king
        rewards[env][player] = (
            ( is_action_ok &  is_king_ok) * +0.0f +
            (!is_action_ok | !is_king_ok) * -1.0f
        );

        // if the player's action left the king uncovered, enemy get +1
        // otherwise nothing
        rewards[env][enemy] = (
            (!is_king_ok) * +1.0f + 
            ( is_king_ok) * +0.0f
        );

        // if one makes an illegal action, or 
        // if one leave the king in check terminate the environment
        dones[env] = !is_action_ok | !is_king_ok;
    }
}



    

