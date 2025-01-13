#pragma once
#include <torch/extension.h>
#include "chess-consts.h"
#include "clamp.cu"


__device__ long count_attacks(
    int env, int row, int col, 
    torch::PackedTensorAccessor64<long , 1, torch::RestrictPtrTraits> players,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> boards
) {
    long attacks = 0;

    // relative pieces
    const short OPPONENT_KNIGHT = ((players[env] + 1) % 2 * 6) + WHITE_KNIGHT;
    const short OPPONENT_BISHOP = ((players[env] + 1) % 2 * 6) + WHITE_BISHOP;
    const short OPPONENT_ROOK   = ((players[env] + 1) % 2 * 6) + WHITE_ROOK;
    const short OPPONENT_QUEEN  = ((players[env] + 1) % 2 * 6) + WHITE_QUEEN;
    const short OPPONENT_KING   = ((players[env] + 1) % 2 * 6) + WHITE_KING;

    // if player is white count attacks by blacks pawns
    attacks += (players[env] == WHITE) * (row > 0) * (col > 0) * (boards[env][clamp(0,63,(row - 1) * 8 + col - 1)] == BLACK_PAWN);
    attacks += (players[env] == WHITE) * (row > 0) * (col < 7) * (boards[env][clamp(0,63,(row - 1) * 8 + col + 1)] == BLACK_PAWN);

    // if player is black count attacks by white pawns
    attacks += (players[env] == BLACK) * (row < 7) * (col > 0) * (boards[env][clamp(0,63,(row + 1) * 8 + col - 1)] == WHITE_PAWN);
    attacks += (players[env] == BLACK) * (row < 7) * (col < 7) * (boards[env][clamp(0,63,(row + 1) * 8 + col + 1)] == WHITE_PAWN);
    
    
    // count knight attacks
    attacks += (row > 1) * (col > 0) * (boards[env][clamp(0,63,(row - 2) * 8 + (col - 1))] == OPPONENT_KNIGHT);
    attacks += (row > 1) * (col < 7) * (boards[env][clamp(0,63,(row - 2) * 8 + (col + 1))] == OPPONENT_KNIGHT);
    attacks += (row > 0) * (col > 1) * (boards[env][clamp(0,63,(row - 1) * 8 + (col - 2))] == OPPONENT_KNIGHT);
    attacks += (row > 0) * (col < 6) * (boards[env][clamp(0,63,(row - 1) * 8 + (col + 2))] == OPPONENT_KNIGHT);
    attacks += (row < 7) * (col > 1) * (boards[env][clamp(0,63,(row + 1) * 8 + (col - 2))] == OPPONENT_KNIGHT);
    attacks += (row < 7) * (col < 6) * (boards[env][clamp(0,63,(row + 1) * 8 + (col + 2))] == OPPONENT_KNIGHT);
    attacks += (row < 6) * (col > 0) * (boards[env][clamp(0,63,(row + 2) * 8 + (col - 1))] == OPPONENT_KNIGHT);
    attacks += (row < 6) * (col < 7) * (boards[env][clamp(0,63,(row + 2) * 8 + (col + 1))] == OPPONENT_KNIGHT);

    
    // count king attacks
    attacks += (row > 0) * (col > 0) * (boards[env][clamp(0,63,(row - 1) * 8 + (col - 1))] == OPPONENT_KING);
    attacks += (row > 0) * (col < 7) * (boards[env][clamp(0,63,(row - 1) * 8 + (col + 1))] == OPPONENT_KING);
    attacks += (row < 7) * (col > 0) * (boards[env][clamp(0,63,(row + 1) * 8 + (col - 1))] == OPPONENT_KING);
    attacks += (row < 7) * (col < 7) * (boards[env][clamp(0,63,(row + 1) * 8 + (col + 1))] == OPPONENT_KING);
    attacks += (row > 0) * (boards[env][clamp(0,63,(row - 1) * 8 + col)] == OPPONENT_KING);
    attacks += (row < 7) * (boards[env][clamp(0,63,(row + 1) * 8 + col)] == OPPONENT_KING);
    attacks += (col > 0) * (boards[env][clamp(0,63,row * 8 + (col - 1))] == OPPONENT_KING);
    attacks += (col < 7) * (boards[env][clamp(0,63,row * 8 + (col + 1))] == OPPONENT_KING);
    
    
    // count bottom-right attacks
    bool covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (row + i < 8) * (col + i < 8) * (boards[env][clamp(0,63,(row + i) * 8 + (col + i))] == OPPONENT_BISHOP || boards[env][clamp(0,63,(row + i) * 8 + (col + i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,(row + i) * 8 + (col + i))] != EMPTY);
    }
    
    // count bottom-left attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (row + i < 8) * (col - i >= 0) * (boards[env][clamp(0,63,(row + i) * 8 + (col - i))] == OPPONENT_BISHOP || boards[env][clamp(0,63,(row + i) * 8 + (col - i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,(row + i) * 8 + (col - i))] != EMPTY);
    }

    // count top-right attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (row - i >= 0) * (col + i < 8) * (boards[env][clamp(0,63,(row - i) * 8 + (col + i))] == OPPONENT_BISHOP || boards[env][clamp(0,63,(row - i) * 8 + (col + i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,(row - i) * 8 + (col + i))] != EMPTY);
    }

    // count top-left attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (row - i >= 0) * (col - i >= 0) * (boards[env][clamp(0,63,(row - i) * 8 + (col - i))] == OPPONENT_BISHOP || boards[env][clamp(0,63,(row - i) * 8 + (col - i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,(row - i) * 8 + (col - i))] != EMPTY);
    }

    // count bottom attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (row + i < 8) * (boards[env][clamp(0,63,(row + i) * 8 + col)] == OPPONENT_ROOK || boards[env][clamp(0,63,(row + i) * 8 + col)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,(row + i) * 8 + col)] != EMPTY);
    }

    // count top attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (row - i >= 0) * (boards[env][clamp(0,63,(row - i) * 8 + col)] == OPPONENT_ROOK || boards[env][clamp(0,63,(row - i) * 8 + col)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,(row - i) * 8 + col)] != EMPTY);
    }

    // count right attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (col + i < 8) * (boards[env][clamp(0,63,row * 8 + col + i)] == OPPONENT_ROOK || boards[env][clamp(0,63,row * 8 + col + i)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,row * 8 + col + i)] != EMPTY);
    }

    // count left attacks
    covered = false;
    for (int i = 1; i < 8; i++) {
        attacks += (!covered) * (col - i >= 0) * (boards[env][clamp(0,63,row * 8 + col - i)] == OPPONENT_ROOK || boards[env][clamp(0,63,row * 8 + col - i)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][clamp(0,63,row * 8 + col - i)] != EMPTY);
    }
    
    return attacks;

}

__global__ void attacks_kernel(
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> boards,
    torch::PackedTensorAccessor64<long , 1, torch::RestrictPtrTraits> players,
    torch::PackedTensorAccessor64<long , 2, torch::RestrictPtrTraits> colors
) {
    const int env = blockIdx.x;
    const int row = threadIdx.y;
    const int col = threadIdx.x;

    colors[env][row * 8 + col] = count_attacks(env, row, col, players, boards);
}
