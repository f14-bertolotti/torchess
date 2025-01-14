#pragma once
#include <torch/extension.h>
#include "chess-consts.h"

__device__ unsigned char standard_action(
    int env,
    torch::PackedTensorAccessor64<long , 1, torch::RestrictPtrTraits> players,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> boards,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> actions
) {
    // performs a standard action
    // returns 0 if everything is ok
    // returns 1 if the action was a standard action but the conditions were not met

    const char PLAYER_PAWN = 6 * players[env] + WHITE_PAWN;

    const bool is_standard_action = (
        (actions[env][0] >= 0) * (actions[env][0] < 8) * // 0 <= action source row <= 7
        (actions[env][1] >= 0) * (actions[env][1] < 8) * // 0 <= action source col <= 7
        (actions[env][2] >= 0) * (actions[env][2] < 8) * // 0 <= action target row <= 7
        (actions[env][3] >= 0) * (actions[env][3] < 8) * // 0 <= action target col <= 7
        (actions[env][4] == 0)                           // no special action
    );

    const bool is_source_fill = boards[actions[env][0] * 8 + actions[env][0]] != EMPTY;
    
    const bool is_pawn_moving = (
        (actions[env][0] * 8 + actions[env][1] == PLAYER_PAWN           ) * // pawn is in the right position
        (actions[env][0] == actions[env][2] (players[env]==BLACK?+1:-1) ) * // pawn is moving one row forward
        (actions[env][1] == actions[env][3]                             ) * // pawn is not changing col
        (boards[env][actions[env][2] * 8 + actions[env][3]] == EMPTY    )   // target is empty
    );
    
    const bool is_pawn_capturing = (
        (boards[env][actions[env][0] * 8 + actions[env][1]] == PLAYER_PAWN   ) * // pawn is in the right position
        (actions[env][0] == actions[env][2] (players[env]==(BLACK ? +1 : -1))) * // pawn is moving one row forward
        (abs(actions[env][1] - actions[env][3]) == 1                         ) * // pawn is moving one col to the side
        (( // target is black or white
            (boards[env][actions[env][2] * 8 + actions[env][3]] > 0  ) *
            (boards[env][actions[env][2] * 8 + actions[env][3]] < 6  )   // target is white
        ) + 
        (
            (boards[env][actions[env][2] * 8 + actions[env][3]] > 6  ) * 
            (boards[env][actions[env][2] * 8 + actions[env][3]] < 12 )   // target is black
        ))
    );

}

__device__ unsigned char kingside_castle(
    int env,
    torch::PackedTensorAccessor64<long , 1, torch::RestrictPtrTraits> players,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> boards ,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> actions
) {
    // performs kingside castling action
    // returns 0 if everything is ok
    // returns 1 if the action was a kingside castling but the conditions were not met

    const unsigned char PLAYER_KING = players[env] * 6 + WHITE_KING;
    const unsigned char PLAYER_ROOK = players[env] * 6 + WHITE_ROOK;
    const unsigned char special = actions[env][4];
    const unsigned char king_source = (7 * players[env]) * 8 + 4;
    const unsigned char rook_source = (7 * players[env]) * 8 + 7;
    const unsigned char king_target = (7 * players[env]) * 8 + 6;
    const unsigned char rook_target = (7 * players[env]) * 8 + 5;

    const bool is_kingside_castle = (
        (actions[env][0] == 0   ) * // action source empty
        (actions[env][1] == 0   ) * // action source empty
        (actions[env][2] == 0   ) * // action target empty
        (actions[env][3] == 0   ) * // action target empty
        (special == KING_CASTLE )   // king castling action
    );

    const bool is_action_ok = ( 
        (boards[env][64 + players[env]] == 0                      ) * // king has not moved
        (boards[env][66 + players[env]] == 0                      ) * // king-side rook has not moved
        (boards[env][king_source] == PLAYER_KING                  ) * // king is in the right position
        (boards[env][rook_target] == EMPTY                        ) * // king-side is empty
        (boards[env][king_target] == EMPTY                        ) * // king-side is empty
        (boards[env][rook_source] == PLAYER_ROOK                  ) * // king-side rook is in the right position
        (count_attacks(env, king_source, 4, players, boards) == 0 ) * // king is not in check
        (count_attacks(env, king_target, 5, players, boards) == 0 ) * // king-side 1 is not in check
        (count_attacks(env, rook_target, 6, players, boards) == 0 )   // king-side 2 is not in check
    ); 

    boards[env][king_source] = is_kingside_castle * is_action_ok ? EMPTY       : boards[env][king_source];
    boards[env][rook_source] = is_kingside_castle * is_action_ok ? EMPTY       : boards[env][rook_source];
    boards[env][rook_target] = is_kingside_castle * is_action_ok ? PLAYER_ROOK : boards[env][rook_target];
    boards[env][king_target] = is_kingside_castle * is_action_ok ? PLAYER_KING : boards[env][king_target];

    return is_kingside_castle * (!is_action_ok);
}

__device__ unsigned char queenside_castle(
    int env,
    torch::PackedTensorAccessor64<long , 1, torch::RestrictPtrTraits> players,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> boards,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> actions
) {
    // performs queenside castling action
    // returns 0 if everything is ok
    // returns 1 if the action was a queenside castling but the conditions were not met

    const unsigned char PLAYER_KING = players[env] * 6 + WHITE_KING;
    const unsigned char PLAYER_ROOK = players[env] * 6 + WHITE_ROOK;
    const unsigned char special = actions[env][4];
    const unsigned char king_source = (7 * players[env]) * 8 + 4;
    const unsigned char rook_target = (7 * players[env]) * 8 + 3;
    const unsigned char king_target = (7 * players[env]) * 8 + 2;
    const unsigned char rook_side   = (7 * players[env]) * 8 + 1;
    const unsigned char rook_source = (7 * players[env]) * 8 + 0;

    const bool is_queenside_castle =  (
        (actions[env][0] == 0    ) * // action source empty
        (actions[env][1] == 0    ) * // action source empty
        (actions[env][2] == 0    ) * // action target empty
        (actions[env][3] == 0    ) * // action target empty
        (special == QUEEN_CASTLE )   // queenside castling action
    );

    const bool is_action_ok = ( 
        (boards[env][64 + players[env]] == 0                      ) * // king has not moved
        (boards[env][68 + players[env]] == 0                      ) * // queen-side rook has not moved
        (boards[env][king_source] == PLAYER_KING                  ) * // king is in the right position
        (boards[env][rook_target] == EMPTY                        ) * // rook-target is empty
        (boards[env][king_target] == EMPTY                        ) * // king-target is empty
        (boards[env][rook_side]   == EMPTY                        ) * // rook-side is empty
        (boards[env][rook_source] == PLAYER_ROOK                  ) * // rook is in the right position
        (count_attacks(env, king_source, 4, players, boards) == 0 ) * // king is not in check
        (count_attacks(env, king_target, 3, players, boards) == 0 ) * // king target is not in check
        (count_attacks(env, rook_target, 2, players, boards) == 0 )   // rook target is not in check
    );

    boards[env][rook_target] = is_queenside_castle * is_action_ok ? PLAYER_ROOK : boards[env][rook_target];
    boards[env][king_target] = is_queenside_castle * is_action_ok ? PLAYER_KING : boards[env][king_target];
    boards[env][king_source] = is_queenside_castle * is_action_ok ? EMPTY       : boards[env][king_source];
    boards[env][rook_side]   = is_queenside_castle * is_action_ok ? EMPTY       : boards[env][rook_side];
    boards[env][rook_source] = is_queenside_castle * is_action_ok ? EMPTY       : boards[env][rook_source];

    return is_queenside_castle * (!is_action_ok);
}

    

__global__ void step_kernel(
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> boards,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> actions,
    torch::PackedTensorAccessor64<long , 1, torch::RestrictPtrTraits> players,
    torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> rewards,
    torch::PackedTensorAccessor64<bool , 1, torch::RestrictPtrTraits> dones
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
