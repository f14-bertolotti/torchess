#pragma once
#include <torch/extension.h>
#include "chess-attacks.cu"
#include "chess-consts.h"
#include <stdio.h>

__device__ unsigned char validate_actions(
    int env,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs a standard action
    // returns 0 if everything is ok
    // returns 1 if the action was a standard action but the conditions were not met

    const unsigned char player_pawn = 6 * players[env] + WHITE_PAWN;
    const unsigned char enemy_pawn  = 6 * (players[env] + 1) % 2 + WHITE_PAWN;
    const unsigned char source_index = actions[env][0] * 8 + actions[env][1];
    const unsigned char target_index = actions[env][2] * 8 + actions[env][3];
    const unsigned char prev_action_source_index = boards[env][PREV_ACTION + 0] * 8 + boards[env][PREV_ACTION + 1];
    const unsigned char prev_action_target_index = boards[env][PREV_ACTION + 2] * 8 + boards[env][PREV_ACTION + 3];
    const unsigned char prev_action_special = boards[env][PREV_ACTION + 4];
    const unsigned char source_piece = boards[env][source_index];
    const unsigned char target_piece = boards[env][target_index];

    const bool is_special_action = (actions[env][4] == 0);
    const bool is_target_empty = (boards[env][target_index] == EMPTY);
    const bool is_promotion = (actions[env][4] > 3 & actions[env][4] < 8);

    const bool are_bounds_ok = (
        (actions[env][0] >= 0) & (actions[env][0] < 8) & // 0 <= action source row <= 7
        (actions[env][1] >= 0) & (actions[env][1] < 8) & // 0 <= action source col <= 7
        (actions[env][2] >= 0) & (actions[env][2] < 8) & // 0 <= action target row <= 7
        (actions[env][3] >= 0) & (actions[env][3] < 8) & // 0 <= action target col <= 7
        (actions[env][4] >= 0)                           // positive
    );

    // source should be of the player's color
    const bool is_source_ok = (
        ( 
            (players[env] == BLACK) & // player is black
            (source_piece >= 7    ) & // source is black
            (source_piece <= 12   )   // source is black
        ) | 
        (
            (players[env] == WHITE ) & // player is white
            (source_piece >= 1     ) & // source is white
            (source_piece <= 6     )   // source is white
        )
    );

    // target should be of the opponent's color
    const bool is_capturing = (
        ( 
            (players[env] == WHITE) & // player is white
            (target_piece >= 7    ) & // target is black
            (target_piece <= 11   )   
        ) | 
        ( 
            (players[env] == BLACK) & // player is black
            (target_piece >= 0    ) & // target is white
            (target_piece <= 5    ) 
        ) 
    );

    const bool is_pawn_source = source_piece == player_pawn;

    const bool is_pawn_moving = (
        are_bounds_ok                                                            & // action is a standard action
        !is_special_action                                                       & // action is not a special action
        is_source_ok                                                             & // source is of the player's color
        is_target_empty                                                          & // target is empty
        is_pawn_source                                                           & // source is a pawn
        (actions[env][2] == (actions[env][0] + (players[env]==BLACK ? +1 : -1))) & // pawn is moving one row forward
        (actions[env][1] == actions[env][3]                                    )   // pawn is not changing col
    );
    
    const bool is_pawn_capturing = (
        are_bounds_ok                                                            & // action is well formed
        !is_special_action                                                       & // action is not a special action
        is_source_ok                                                             & // source is of the player's color
        is_capturing                                                             & // target is of the opponent's color
        is_pawn_source                                                           & // source is a pawn
        (actions[env][2] == (actions[env][0] + (players[env]==BLACK ? +1 : -1))) & // pawn is moving one row forward
        (abs(actions[env][1] - actions[env][3]) == 1                           )   // pawn is moving one col to the side
    );

    const bool is_pawn_en_passant = (
        are_bounds_ok                                                            & // action is well formed
        !is_special_action                                                       & // action is not a special action
        is_source_ok                                                             & // source is of the player's color
        is_pawn_source                                                           & // source is a pawn
        (actions[env][2] == (actions[env][0] + (players[env]==BLACK ? +1 : -1))) & // pawn is moving one row forward
        (abs(actions[env][1] - actions[env][3]) == 1                           ) & // pawn is moving one col to the side
        (boards[env][actions[env][0] * 8 + actions[env][3]] == enemy_pawn      ) & // has an enemy pawn on the side of the movement
        (prev_action_target_index == actions[env][0] * 8 + actions[env][1]     ) & // the enemy pawn moved in the previous action
        (abs(boards[env][PREV_ACTION + 0] - boards[env][PREV_ACTION + 2]) == 2 )   // the enemy pawn moved two rows
    );

    const bool is_pawn_double_move = (
        are_bounds_ok                                                            & // action is well formed
        !is_special_action                                                       & // action is not a special action
        is_source_ok                                                             & // source is of the player's color
        is_target_empty                                                          & // target is empty
        is_pawn_source                                                           & // source is a pawn
        (actions[env][2] == (actions[env][0] + (players[env]==BLACK ? +2 : -2))) & // pawn is moving two rows forward
        (actions[env][1] == actions[env][3]                                    ) & // pawn is not changing col
        (boards[env][(actions[env][0] + (players[env]==BLACK ? +1 : -1)) * 8 + actions[env][1]] == EMPTY ) & // the row in between is empty
        (boards[env][(actions[env][0] + (players[env]==BLACK ? +2 : -2)) * 8 + actions[env][1]] == EMPTY )   // the row in between is empty
    );
    
    const bool is_pawn_promotion = (
        are_bounds_ok                                                            & // action is well formed
        is_special_action                                                        & // action is not a special action
        is_source_ok                                                             & // source is of the player's color
        is_target_empty                                                          & // target is empty
        is_pawn_source                                                           & // source is a pawn
        (actions[env][2] == (players[env]==BLACK ? 0 : 7)                      ) & // pawn is moving to the last row
        (actions[env][2] == (actions[env][0] + (players[env]==BLACK ? +1 : -1))) & // pawn is moving one row forward
        (actions[env][1] == actions[env][3]                                    )   // pawn is not changing col
    );

    return (
        is_pawn_moving      * 1 +
        is_pawn_capturing   * 2 +
        is_pawn_en_passant  * 3 +
        is_pawn_double_move * 4 +
        is_pawn_promotion   * 5
    );

}



    

__global__ void step_kernel(
    torch::PackedTensorAccessor32<int   , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int   , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int   , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<float , 2 , torch::RestrictPtrTraits> rewards ,
    torch::PackedTensorAccessor32<bool  , 1 , torch::RestrictPtrTraits> dones
) {
    // This function given the action and the current boards computes the next state of the boards
    // It also updates the rewards and the dones
    // The function is in-place, it modifies input/actions/dones/rewards in place

    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned char OPPONENT_PAWN   = ((players[env] + 1) % 2 * 6) + WHITE_PAWN;
    const unsigned char OPPONENT_KNIGHT = ((players[env] + 1) % 2 * 6) + WHITE_KNIGHT;
    const unsigned char OPPONENT_BISHOP = ((players[env] + 1) % 2 * 6) + WHITE_BISHOP;
    const unsigned char OPPONENT_ROOK   = ((players[env] + 1) % 2 * 6) + WHITE_ROOK;
    const unsigned char OPPONENT_QUEEN  = ((players[env] + 1) % 2 * 6) + WHITE_QUEEN;
    const unsigned char OPPONENT_KING   = ((players[env] + 1) % 2 * 6) + WHITE_KING;

    const unsigned char PLAYER_PAWN   = players[env] * 6 + WHITE_PAWN;
    const unsigned char PLAYER_KNIGHT = players[env] * 6 + WHITE_KNIGHT;
    const unsigned char PLAYER_BISHOP = players[env] * 6 + WHITE_BISHOP;
    const unsigned char PLAYER_ROOK   = players[env] * 6 + WHITE_ROOK;
    const unsigned char PLAYER_QUEEN  = players[env] * 6 + WHITE_QUEEN;
    const unsigned char PLAYER_KING   = players[env] * 6 + WHITE_KING;

    const bool  WHITE_TURN = players[0] == 0;
    const bool  BLACK_TURN = players[0] == 1;

    const unsigned char src = actions[env][0] * 8 + actions[env][1];
    const unsigned char tgt = actions[env][2] * 8 + actions[env][3];
    const unsigned char spc = actions[env][4];

    const unsigned char  kingside_castling_result =  kingside_castle(env, players, boards, actions);
    const unsigned char queenside_castling_result = queenside_castle(env, players, boards, actions);

    // // check if action indexes are inside the board
    // if (actions[env][0] < 0 || actions[env][0] >= 8 || // starting row
    //     actions[env][1] < 0 || actions[env][1] >= 8 || // starting col
    //     actions[env][2] < 0 || actions[env][2] >= 8 || // ending row
    //     actions[env][3] < 0 || actions[env][3] >= 8    // ending col
    // ) {
    //     dones[env] = true;
    //     rewards[env][players[env]] = -1;
    //     return;
    // }

    // // check starting position is not empty
    // if (boards[env][src] == 0) {
    //     dones[env] = true;
    //     rewards[env][players[env]] = -1;
    //     return;
    // }

    // // check if the player is moving his own piece
    // if (boards[env][src] > players[env] * 6 && 
    //     boards[env][src] <= players[env] * 6 + 6
    // ) {
    //     dones[env] = true;
    //     rewards[env][players[env]] = -1;
    //     return;
    // }


}




