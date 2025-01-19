#include <torch/extension.h>
#include "chess-attacks.cu"
#include "moves/kingside-castling.cu"
#include "moves/queenside-castling.cu"
#include "moves/promotion.cu"
#include "moves/pawn-move.cu"
#include "moves/double-move.cu"
#include "moves/en-passant.cu"
#include "moves/knight-move.cu"
#include "moves/king-move.cu"

/*torch::Tensor step(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor rewards, torch::Tensor dones) {
    // The sole purpose of this function is to check inputs shapes, and launch the kernel

    // assume boards shape is (N, 100)
    if (boards.dim()   != 2  ) throw std::invalid_argument("Boards tensor must be 3D, (N, 132)");
    if (boards.size(1) != 100) throw std::invalid_argument("First dimension must be 100, found " + std::to_string(boards.size(1)));

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
        boards    .packed_accessor32<int   , 2 , torch::RestrictPtrTraits>() ,
        actions   .packed_accessor32<int   , 2 , torch::RestrictPtrTraits>() ,
        players   .packed_accessor32<int   , 1 , torch::RestrictPtrTraits>() ,
        rewards   .packed_accessor32<float , 2 , torch::RestrictPtrTraits>() ,
        dones     .packed_accessor32<bool  , 1 , torch::RestrictPtrTraits>()
    );

    return boards;
}
*/

void kingside_castling(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    kingside_castle_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void queenside_castling(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    queenside_castle_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void promotion(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    promotion_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void pawn_move(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    pawn_move_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void knight_move(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    knight_move_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void king_move(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    king_move_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void pawn_double(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    doublemove_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void pawn_en_passant(torch::Tensor boards, torch::Tensor actions, torch::Tensor players, torch::Tensor result) {
    int threads = 128;
    int blocks = (boards.size(0) + threads - 1) / threads;
    en_passant_kernel<<<blocks, threads>>>(
        boards .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        actions.packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players.packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result .packed_accessor32<int , 1 , torch::RestrictPtrTraits>()
    );
}

void attacks(torch::Tensor boards, torch::Tensor players, torch::Tensor result) {
    // The sole purpose of this function is to make sanity cheks and launch the kernel

    // assume boards shape is (N, 100)
    TORCH_CHECK(boards.dim()   == 2 , "Boards tensor must be 3D, (N, 100)");
    TORCH_CHECK(boards.size(1) == 100, "First dimension must be 100, found " + std::to_string(boards.size(1)));

    // assume colors shape is (N, 64)
    TORCH_CHECK(result.dim()   == 2 , "Colors tensor must be 2D, (N, 64)");
    TORCH_CHECK(result.size(1) == 64, "First dimension must be 64, found " + std::to_string(result.size(1)));

    // assume players shape is (N)
    TORCH_CHECK(players.dim() == 1, "Players tensor must be 1D, (N)");

    // all tensor must be on gpu
    TORCH_CHECK(boards .is_cuda(),  "boards must be a CUDA tensor");
    TORCH_CHECK(players.is_cuda(), "players must be a CUDA tensor");
    TORCH_CHECK(result .is_cuda(),  "colors must be a CUDA tensor");

    // launch a 64-threads-block for each board
    dim3 griddim(boards.size(0));
    dim3 blockdim(8, 8);
    attacks_kernel<<<griddim, blockdim>>>(
        boards    .packed_accessor32<int , 2 , torch::RestrictPtrTraits>() ,
        players   .packed_accessor32<int , 1 , torch::RestrictPtrTraits>() ,
        result    .packed_accessor32<int , 2 , torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();

    // check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

// macro to create the python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, python_module) {
    python_module.def("attacks"            , &attacks            , "count attacks in the board" );
    python_module.def("kingside_castling"  , &kingside_castling  , "kingside castling action"   );
    python_module.def("queenside_castling" , &queenside_castling , "queenside castling action"  );
    python_module.def("promotion"          , &promotion          , "pawn promotion action"      );
    python_module.def("pawn_move"          , &pawn_move          , "pawn move action"           );
    python_module.def("pawn_double"        , &pawn_double        , "pawn double move action"    );
    python_module.def("pawn_en_passant"    , &pawn_en_passant    , "pawn en passant action"     );
    python_module.def("knight_move"        , &knight_move        , "knight move action"         );
    python_module.def("king_move"          , &king_move          , "king move action"           );
}
