#include <torch/extension.h>

const int EMPTY = 0;
const int WHITE_PAWN   = 1;
const int WHITE_KNIGHT = 2;
const int WHITE_BISHOP = 3;
const int WHITE_ROOK   = 4;
const int WHITE_QUEEN  = 5;
const int WHITE_KING   = 6;
const int BLACK_PAWN   = 7;
const int BLACK_KNIGHT = 8;
const int BLACK_BISHOP = 9;
const int BLACK_ROOK   = 10;
const int BLACK_QUEEN  = 11;
const int BLACK_KING   = 12;
const int WHITE  = 0;
const int BLACK  = 1;

/*
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

    // const short PLAYER_PAWN     = (players[0] * 6) + PAWN;
    // const short PLAYER_KNIGHT   = (players[0] * 6) + KNIGHT;
    // const short PLAYER_BISHOP   = (players[0] * 6) + BISHOP;
    // const short PLAYER_ROOK     = (players[0] * 6) + ROOK;
    // const short PLAYER_QUEEN    = (players[0] * 6) + QUEEN;
    // const short PLAYER_KING     = (players[0] * 6) + KING;
    // const short OPPONENT_PAWN   = ((players[0] + 1) % 2 * 6) + PAWN;
    // const short OPPONENT_KNIGHT = ((players[0] + 1) % 2 * 6) + KNIGHT;
    // const short OPPONENT_BISHOP = ((players[0] + 1) % 2 * 6) + BISHOP;
    // const short OPPONENT_ROOK   = ((players[0] + 1) % 2 * 6) + ROOK;
    // const short OPPONENT_QUEEN  = ((players[0] + 1) % 2 * 6) + QUEEN;
    // const short OPPONENT_KING   = ((players[0] + 1) % 2 * 6) + KING;
    // const bool  WHITE_TURN      = players[0] == 0;
    // const bool  BLACK_TURN      = players[0] == 1;

    // int env = blockIdx.x * blockDim.x + threadIdx.x;
    // int src = actions[env][0] * 8 + actions[env][1];
    // int tgt = actions[env][2] * 8 + actions[env][3];
    // int spc = actions[env][4];

    // // king castling
    // if (spc == 1                                                      && // king castling action
    //     boards[env][64 + players[env]] == 0                           && // king has not moved
    //     boards[env][66 + players[env]] == 0                           && // king-side rook has not moved
    //     boards[env][7 * players[env] * 8 + 4] == players[env] * 6 + 5 && // king is there
    //     boards[env][7 * players[env] * 8 + 5] == 0                    && // king-side is empty
    //     boards[env][7 * players[env] * 8 + 6] == 0                    && // king-side is empty
    //     boards[env][7 * players[env] * 8 + 7] == players[env] * 6 + 2    // king-side rook is there
    // ) {
    //     boards[env][7 * players[env] * 8 + 4] = players[env] * 6 + 2;
    //     boards[env][7 * players[env] * 8 + 7] = players[env] * 6 + 5;
    // }

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

torch::Tensor step(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor rewards, torch::Tensor dones) {
    // The sole purpose of this function is to check inputs shapes, and launch the kernel

    // assume boards shape is (N, 68)
    if (boards.dim()   != 2  ) throw std::invalid_argument("Boards tensor must be 3D, (N, 132)");
    if (boards.size(1) != 132) throw std::invalid_argument("First dimension must be 132, found " + std::to_string(boards.size(1)));

    // assume actions shape is (N, 4)
    if (actions.dim()   != 2) throw std::invalid_argument("Actions tensor must be 2D, (N, 4)");
    if (actions.size(1) != 4) throw std::invalid_argument("First dimension must be 4, found " + std::to_string(actions.size(1)));

    // assume players shape is (N)
    if (players.dim() != 1) throw std::invalid_argument("Players tensor must be 1D, (N)");

    // assume rewards shape is (N,2)
    if (rewards.dim() != 2) throw std::invalid_argument("Rewards tensor must be 2D, (N, 2)");
    if (rewards.size(1) != 2) throw std::invalid_argument("First dimension must be 2, found " + std::to_string(rewards.size(1)));

    // assume terminated shape is (N)
    if (dones.dim() != 1) throw std::invalid_argument("Dones tensor must be 1D, (N)");

    // zero-fill rewards and dones
    rewards.fill_(0);
    dones.fill_(false);

    // launch the necessary block made of 1024 threads
    int threads = 1024;
    int blocks = (boards.size(0) + threads - 1) / threads;
    step_kernel<<<blocks, threads>>>(
        boards    .packed_accessor64<long , 2, torch::RestrictPtrTraits>(),
        actions   .packed_accessor64<long , 2, torch::RestrictPtrTraits>(),
        players   .packed_accessor64<long , 1, torch::RestrictPtrTraits>(),
        rewards   .packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        dones     .packed_accessor64<bool , 1, torch::RestrictPtrTraits>()
    );

    return boards;
}*/

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
    attacks += (players[env] == WHITE) * (row > 0) * (col > 0) * (boards[env][(row > 0) * (col > 0) * ((row - 1) * 8 + col - 1)] == BLACK_PAWN);
    attacks += (players[env] == WHITE) * (row > 0) * (col < 7) * (boards[env][(row > 0) * (col < 7) * ((row - 1) * 8 + col + 1)] == BLACK_PAWN);

    // if player is black count attacks by white pawns
    attacks += (players[env] == BLACK) * (row < 7) * (col > 0) * (boards[env][(row < 7) * (col > 0) * ((row + 1) * 8 + col - 1)] == WHITE_PAWN);
    attacks += (players[env] == BLACK) * (row < 7) * (col < 7) * (boards[env][(row < 7) * (col < 7) * ((row + 1) * 8 + col + 1)] == WHITE_PAWN);
    
    
    // count knight attacks
    attacks += (row > 1) * (col > 0) * (boards[env][(row > 1) * (col > 0) * ((row - 2) * 8 + (col - 1))] == OPPONENT_KNIGHT);
    attacks += (row > 1) * (col < 7) * (boards[env][(row > 1) * (col < 7) * ((row - 2) * 8 + (col + 1))] == OPPONENT_KNIGHT);
    attacks += (row > 0) * (col > 1) * (boards[env][(row > 0) * (col > 1) * ((row - 1) * 8 + (col - 2))] == OPPONENT_KNIGHT);
    attacks += (row > 0) * (col < 6) * (boards[env][(row > 0) * (col < 6) * ((row - 1) * 8 + (col + 2))] == OPPONENT_KNIGHT);
    attacks += (row < 7) * (col > 1) * (boards[env][(row < 7) * (col > 1) * ((row + 1) * 8 + (col - 2))] == OPPONENT_KNIGHT);
    attacks += (row < 7) * (col < 6) * (boards[env][(row < 7) * (col < 6) * ((row + 1) * 8 + (col + 2))] == OPPONENT_KNIGHT);
    attacks += (row < 6) * (col > 0) * (boards[env][(row < 6) * (col > 0) * ((row + 2) * 8 + (col - 1))] == OPPONENT_KNIGHT);
    attacks += (row < 6) * (col < 7) * (boards[env][(row < 6) * (col < 7) * ((row + 2) * 8 + (col + 1))] == OPPONENT_KNIGHT);

    
    // count king attacks
    attacks += (row > 0) * (col > 0) * (boards[env][(row > 0) * (col > 0) * ((row - 1) * 8 + (col - 1))] == OPPONENT_KING);
    attacks += (row > 0) * (col < 7) * (boards[env][(row > 0) * (col < 7) * ((row - 1) * 8 + (col + 1))] == OPPONENT_KING);
    attacks += (row < 7) * (col > 0) * (boards[env][(row < 7) * (col > 0) * ((row + 1) * 8 + (col - 1))] == OPPONENT_KING);
    attacks += (row < 7) * (col < 7) * (boards[env][(row < 7) * (col < 7) * ((row + 1) * 8 + (col + 1))] == OPPONENT_KING);
    attacks += (row > 0) * (boards[env][(row > 0) * ((row - 1) * 8 + col)] == OPPONENT_KING);
    attacks += (row < 7) * (boards[env][(row < 7) * ((row + 1) * 8 + col)] == OPPONENT_KING);
    attacks += (col > 0) * (boards[env][(col > 0) * (row * 8 + (col - 1))] == OPPONENT_KING);
    attacks += (col < 7) * (boards[env][(col < 7) * (row * 8 + (col + 1))] == OPPONENT_KING);
    
    
    // count bottom-right attacks
    bool covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (row + i < 8) * (col + i < 8) * (boards[env][(row + i < 8) * (col + i < 8) * ((row + i) * 8 + (col + i))] == OPPONENT_BISHOP || boards[env][(row + i < 8) * (col + i < 8) * ((row + i) * 8 + (col + i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(row + i < 8) * (col + i < 8) * ((row + i) * 8 + (col + i))] != EMPTY);
    }
    
    // count bottom-left attacks
    covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (row + i < 8) * (col - i >= 0) * (boards[env][(row + i < 8) * (col - i >= 0) * ((row + i) * 8 + (col - i))] == OPPONENT_BISHOP || boards[env][(row + i < 8) * (col - i >= 0) * ((row + i) * 8 + (col - i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(row + i < 8) * (col - i >= 0) * ((row + i) * 8 + (col - i))] != EMPTY);
    }
   //// 
    // count top-right attacks
    covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (row - i >= 0) * (col + i < 8) * (boards[env][(row - i >= 0) * (col + i < 8) * ((row - i) * 8 + (col + i))] == OPPONENT_BISHOP || boards[env][(row - i >= 0) * (col + i < 8) * ((row - i) * 8 + (col + i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(row - i >= 0) * (col + i < 8) * ((row - i) * 8 + (col + i))] != EMPTY);
    }

    // count top-left attacks
    covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (row - i >= 0) * (col - i >= 0) * (boards[env][(row - i >= 0) * (col - i >= 0) * ((row - i) * 8 + (col - i))] == OPPONENT_BISHOP || boards[env][(row - i >= 0) * (col - i >= 0) * ((row - i) * 8 + (col - i))] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(row - i >= 0) * (col - i >= 0) * ((row - i) * 8 + (col - i))] != EMPTY);
    }

    // count bottom attacks
    covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (row + i < 8) * (boards[env][(row + i < 8) * ((row + i) * 8 + col)] == OPPONENT_ROOK || boards[env][(row + i < 8) * ((row + i) * 8 + col)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(row + i < 8) * ((row + i) * 8 + col)] != EMPTY);
    }

    // count top attacks
    covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (row - i >= 0) * (boards[env][(row - i >= 0) * ((row - i) * 8 + col)] == OPPONENT_ROOK || boards[env][(row - i >= 0) * ((row - i) * 8 + col)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(row - i >= 0) * ((row - i) * 8 + col)] != EMPTY);
    }

    // count right attacks
    covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (col + i < 8) * (boards[env][(col + i < 8) * (row * 8 + col + i)] == OPPONENT_ROOK || boards[env][(col + i < 8) * (row * 8 + col + i)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(col + i < 8) * (row * 8 + col + i)] != EMPTY);
    }

    // count left attacks
    covered = false;
    for (int i = 1; i < 7; i++) {
        attacks += (!covered) * (col - i >= 0) * (boards[env][(col - i >= 0) * (row * 8 + col - i)] == OPPONENT_ROOK || boards[env][(col - i >= 0) * (row * 8 + col - i)] == OPPONENT_QUEEN);
        covered = covered || (boards[env][(col - i >= 0) * (row * 8 + col - i)] != EMPTY);
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

void attacks(torch::Tensor boards, torch::Tensor players, torch::Tensor colors) {
    // The sole purpose of this function is to make sanity cheks and launch the kernel

    // assume boards shape is (N, 68)
    TORCH_CHECK(boards.dim()   == 2 , "Boards tensor must be 3D, (N, 66)");
    TORCH_CHECK(boards.size(1) == 66, "First dimension must be 66, found " + std::to_string(boards.size(1)));

    // assume colors shape is (N, 64)
    TORCH_CHECK(colors.dim()   == 2 , "Colors tensor must be 2D, (N, 64)");
    TORCH_CHECK(colors.size(1) == 64, "First dimension must be 64, found " + std::to_string(colors.size(1)));

    // assume players shape is (N)
    TORCH_CHECK(players.dim() == 1, "Players tensor must be 1D, (N)");

    // all tensor mush be on gpu
    TORCH_CHECK(boards.is_cuda(), "boards must be a CUDA tensor");
    TORCH_CHECK(players.is_cuda(), "players must be a CUDA tensor");
    TORCH_CHECK(colors.is_cuda(), "colors must be a CUDA tensor");

    // launch a 64-threads-block for each board
    dim3 griddim(boards.size(0));
    dim3 blockdim(8, 8);
    attacks_kernel<<<griddim, blockdim>>>(
        boards    .packed_accessor64<long , 2, torch::RestrictPtrTraits>(),
        players   .packed_accessor64<long , 1, torch::RestrictPtrTraits>(),
        colors    .packed_accessor64<long , 2, torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();

    // check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

// macro to create the python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, python_module) {
    //python_module.def("step", &step, "In-place Step function");
    python_module.def("attacks", &attacks, "Color function");
}
