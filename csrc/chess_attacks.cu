#pragma once
#include <torch/extension.h>
#include "chess_consts.h"
#include "clamp.cu"

__device__ unsigned char count_attacks(
    size_t env, unsigned char row, unsigned char col, 
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards
) {
    long attacks = 0;

    // relative pieces
    const unsigned char enemy_knight = ((players[env] + 1) % 2 * 6) + WHITE_KNIGHT;
    const unsigned char enemy_bishop = ((players[env] + 1) % 2 * 6) + WHITE_BISHOP;
    const unsigned char enemy_rook   = ((players[env] + 1) % 2 * 6) + WHITE_ROOK;
    const unsigned char enemy_queen  = ((players[env] + 1) % 2 * 6) + WHITE_QUEEN;
    const unsigned char enemy_king   = ((players[env] + 1) % 2 * 6) + WHITE_KING;

    // if player is white count attacks by blacks pawns
    attacks += (players[env] == WHITE) & (row > 0) & (col > 0) & (boards[env][clamp(0,63,(row - 1) * 8 + col - 1)] == BLACK_PAWN);
    attacks += (players[env] == WHITE) & (row > 0) & (col < 7) & (boards[env][clamp(0,63,(row - 1) * 8 + col + 1)] == BLACK_PAWN);

    // if player is black count attacks by white pawns
    attacks += (players[env] == BLACK) & (row < 7) & (col > 0) & (boards[env][clamp(0,63,(row + 1) * 8 + col - 1)] == WHITE_PAWN);
    attacks += (players[env] == BLACK) & (row < 7) & (col < 7) & (boards[env][clamp(0,63,(row + 1) * 8 + col + 1)] == WHITE_PAWN);
    
    // count knight attacks
    attacks += (row > 1) & (col > 0) & (boards[env][clamp(0,63,(row - 2) * 8 + (col - 1))] == enemy_knight);
    attacks += (row > 1) & (col < 7) & (boards[env][clamp(0,63,(row - 2) * 8 + (col + 1))] == enemy_knight);
    attacks += (row > 0) & (col > 1) & (boards[env][clamp(0,63,(row - 1) * 8 + (col - 2))] == enemy_knight);
    attacks += (row > 0) & (col < 6) & (boards[env][clamp(0,63,(row - 1) * 8 + (col + 2))] == enemy_knight);
    attacks += (row < 7) & (col > 1) & (boards[env][clamp(0,63,(row + 1) * 8 + (col - 2))] == enemy_knight);
    attacks += (row < 7) & (col < 6) & (boards[env][clamp(0,63,(row + 1) * 8 + (col + 2))] == enemy_knight);
    attacks += (row < 6) & (col > 0) & (boards[env][clamp(0,63,(row + 2) * 8 + (col - 1))] == enemy_knight);
    attacks += (row < 6) & (col < 7) & (boards[env][clamp(0,63,(row + 2) * 8 + (col + 1))] == enemy_knight);
    
    // count king attacks
    attacks += (row > 0) & (col > 0) & (boards[env][clamp(0,63,(row - 1) * 8 + (col - 1))] == enemy_king);
    attacks += (row > 0) & (col < 7) & (boards[env][clamp(0,63,(row - 1) * 8 + (col + 1))] == enemy_king);
    attacks += (row < 7) & (col > 0) & (boards[env][clamp(0,63,(row + 1) * 8 + (col - 1))] == enemy_king);
    attacks += (row < 7) & (col < 7) & (boards[env][clamp(0,63,(row + 1) * 8 + (col + 1))] == enemy_king);
    attacks += (row > 0) & (boards[env][clamp(0,63,(row - 1) * 8 + col)] == enemy_king);
    attacks += (row < 7) & (boards[env][clamp(0,63,(row + 1) * 8 + col)] == enemy_king);
    attacks += (col > 0) & (boards[env][clamp(0,63,row * 8 + (col - 1))] == enemy_king);
    attacks += (col < 7) & (boards[env][clamp(0,63,row * 8 + (col + 1))] == enemy_king);
    
    
    // count bottom-right attacks
    bool covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (row + i < 8) & (col + i < 8) & (boards[env][clamp(0,63,(row + i) * 8 + (col + i))] == enemy_bishop | boards[env][clamp(0,63,(row + i) * 8 + (col + i))] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,(row + i) * 8 + (col + i))] != EMPTY);
    }
    
    // count bottom-left attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (row + i < 8) & (col - i >= 0) & (boards[env][clamp(0,63,(row + i) * 8 + (col - i))] == enemy_bishop | boards[env][clamp(0,63,(row + i) * 8 + (col - i))] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,(row + i) * 8 + (col - i))] != EMPTY);
    }

    // count top-right attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (row - i >= 0) & (col + i < 8) & (boards[env][clamp(0,63,(row - i) * 8 + (col + i))] == enemy_bishop | boards[env][clamp(0,63,(row - i) * 8 + (col + i))] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,(row - i) * 8 + (col + i))] != EMPTY);
    }

    // count top-left attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (row - i >= 0) & (col - i >= 0) & (boards[env][clamp(0,63,(row - i) * 8 + (col - i))] == enemy_bishop | boards[env][clamp(0,63,(row - i) * 8 + (col - i))] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,(row - i) * 8 + (col - i))] != EMPTY);
    }

    // count bottom attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (row + i < 8) & (boards[env][clamp(0,63,(row + i) * 8 + col)] == enemy_rook | boards[env][clamp(0,63,(row + i) * 8 + col)] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,(row + i) * 8 + col)] != EMPTY);
    }

    // count top attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (row - i >= 0) & (boards[env][clamp(0,63,(row - i) * 8 + col)] == enemy_rook | boards[env][clamp(0,63,(row - i) * 8 + col)] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,(row - i) * 8 + col)] != EMPTY);
    }

    // count right attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (col + i < 8) & (boards[env][clamp(0,63,row * 8 + col + i)] == enemy_rook | boards[env][clamp(0,63,row * 8 + col + i)] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,row * 8 + col + i)] != EMPTY);
    }

    // count left attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) & (col - i >= 0) & (boards[env][clamp(0,63,row * 8 + col - i)] == enemy_rook | boards[env][clamp(0,63,row * 8 + col - i)] == enemy_queen);
        covered = covered | (boards[env][clamp(0,63,row * 8 + col - i)] != EMPTY);
    }
    
    return attacks * (row <= 7 & col <= 7);

}

__global__ void attacks_kernel(
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> boards  ,
    torch::PackedTensorAccessor32<int , 1 , torch::RestrictPtrTraits> players ,
    torch::PackedTensorAccessor32<int , 2 , torch::RestrictPtrTraits> colors
) {
    const size_t env = blockIdx.x;
    const unsigned char row = threadIdx.y;
    const unsigned char col = threadIdx.x;

    colors[env][row * 8 + col] = count_attacks(env, row, col, players, boards);
}
