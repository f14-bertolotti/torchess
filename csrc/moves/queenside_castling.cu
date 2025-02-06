#pragma once
#include <torch/extension.h>
#include "../chess_attacks.cu"
#include "../chess_consts.h"


__device__ bool queenside_castle_move(
    size_t env,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions
) {
    // performs queenside castling action
    // returns 0 if everything is ok
    // returns 1 if the action was a queenside castling but the conditions were not met

    const unsigned char PLAYER_KING = boards[TURN][env] * 6 + WHITE_KING;
    const unsigned char PLAYER_ROOK = boards[TURN][env] * 6 + WHITE_ROOK;
    const unsigned char castle_row  = boards[TURN][env] == WHITE ? 7 : 0;
    const unsigned char king_source = castle_row * 8 + 4;
    const unsigned char rook_target = castle_row * 8 + 3;
    const unsigned char king_target = castle_row * 8 + 2;
    const unsigned char rook_side   = castle_row * 8 + 1;
    const unsigned char rook_source = castle_row * 8 + 0;
    const unsigned char special = actions[4][env];

    const bool is_queenside_castle =  (
        (actions[0][env] == 0    ) & // action source empty
        (actions[1][env] == 0    ) & // action source empty
        (actions[2][env] == 0    ) & // action target empty
        (actions[3][env] == 0    ) & // action target empty
        (special == QUEEN_CASTLE )   // queenside castling action
    );

    const bool is_action_ok = ( 
        is_queenside_castle                                         & // queenside castling action
        (boards[KING_MOVED + boards[TURN][env]][env] == 0              ) & // king has not moved
        (boards[QUEENSIDE_ROOK_MOVED + boards[TURN][env]][env] == 0    ) & // queen-side rook has not moved
        (boards[king_source][env] == PLAYER_KING                  ) & // king is in the right position
        (boards[rook_target][env] == EMPTY                        ) & // rook-target is empty
        (boards[king_target][env] == EMPTY                        ) & // king-target is empty
        (boards[rook_side][env]   == EMPTY                        ) & // rook-side is empty
        (boards[rook_source][env] == PLAYER_ROOK                  ) & // rook is in the right position
        (count_attacks(env, castle_row, 4, boards) == 0  ) & // king is not in check
        (count_attacks(env, castle_row, 3, boards) == 0  ) & // king target is not in check
        (count_attacks(env, castle_row, 2, boards) == 0  )   // rook target is not in check
    );

    boards[rook_target][env] = is_action_ok ? PLAYER_ROOK : boards[rook_target][env];
    boards[king_target][env] = is_action_ok ? PLAYER_KING : boards[king_target][env];
    boards[king_source][env] = is_action_ok ? EMPTY       : boards[king_source][env];
    boards[rook_side  ][env] = is_action_ok ? EMPTY       : boards[rook_side  ][env];
    boards[rook_source][env] = is_action_ok ? EMPTY       : boards[rook_source][env];

    // update king stored position
    boards[KING_POSITION + boards[TURN][env] * 2 + 0][env] = is_action_ok ? castle_row : boards[KING_POSITION + boards[TURN][env] * 2 + 0][env];
    boards[KING_POSITION + boards[TURN][env] * 2 + 1][env] = is_action_ok ? 4          : boards[KING_POSITION + boards[TURN][env] * 2 + 1][env];

    return !is_action_ok;
}

__global__ void queenside_castle_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> actions ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> result
) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < boards.size(1)) result[env] = queenside_castle_move(env, boards, actions);
}


