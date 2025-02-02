
# Torchess  

**Torchess** is a pure CUDA-based PyTorch extension for chess, designed for reinforcement learning applications. It provides a minimal, low-level interface for managing chess environments efficiently on the GPU.  

## Features  
- Fully implemented in CUDA for fast execution.  
- Simple and minimalistic interface.  
- Designed for reinforcement learning environments.  
- Supports batched operations across multiple environments.  

---

## Installation  

Create a virtual environment and install the package:  
```bash
python3 -m venv venv
source venv/bin/activate
pip install torchess
```

---

## API  

### 1. **Initialization**  
```python
boards, players = init(envs: int)
```
- **Input:**  
  - `envs`: Number of environments (games) to initialize.  
- **Output:**  
  - `boards`: A tensor of shape `(envs, 100)` representing chess boards initialized to the starting position.  
  - `players`: A tensor of shape `(envs,)` indicating the current player to move (0 for white).  

### 2. **Reset**  
```python
boards, players = reset(boards: torch.Tensor, players: torch.Tensor, mask: torch.Tensor | None = None)
```
- **Input:**  
  - `boards`: Current board states.  
  - `players`: Current players.  
  - `mask` (optional): Boolean tensor indicating which environments should be reset.  
- **Output:**  
  - Resets selected boards to the starting position. If `mask` is `None`, all boards are reset.  

### 3. **Step**  
```python
step(boards: torch.Tensor, actions: torch.Tensor, players: torch.Tensor, dones: torch.Tensor, rewards: torch.Tensor)
```
- **Input:**  
  - `boards`: Current board states.  
  - `actions`: Tensor of moves to apply.  
  - `players`: Current players.  
  - `dones`: Tensor indicating which environments have ended.  
  - `rewards`: Tensor storing rewards.  
- **Effect:**  
  - Advances each game by one step.  
  - Updates `boards`, `players`, `dones`, and `rewards` **in place**.  

> **Note:** Any additional structure (e.g., Gym compatibility) should be implemented separately. This is a low-level interface for RL applications.  

---

## Board Representation  

A chessboard is represented as a `100`-element tensor:  
- **First 64 elements**: Piece positions (8×8 board).  
- **Last 36 elements**: Additional game state information.  

### **Piece Encoding**  

| Piece            | Value |
|-----------------|------:|
| Empty           | 0     |
| White Pawn      | 1     |
| White Knight    | 2     |
| White Bishop    | 3     |
| White Rook      | 4     |
| White Queen     | 5     |
| White King      | 6     |
| Black Pawn      | 7     |
| Black Knight    | 8     |
| Black Bishop    | 9     |
| Black Rook      | 10    |
| Black Queen     | 11    |
| Black King      | 12    |

### **Game State Encoding (Last 36 Elements)**  

| Description                      | Index |
|----------------------------------|------:|
| White king moved                | 64    |
| Black king moved                | 65    |
| White kingside rook moved        | 66    |
| Black kingside rook moved        | 67    |
| White queenside rook moved       | 68    |
| Black queenside rook moved       | 69    |
| White previous move              | 70    |
| White move before that           | 75    |
| Black previous move              | 80    |
| Black move before that           | 85    |
| White king position              | 90    |
| Black king position              | 92    |
| Rule 50 counter                  | 94    |
| Threefold repetition counter     | 95    |

---

## Action Representation  

Actions are encoded using **algebraic notation**, specifying:  
1. **Source position** (`row, col`)  
2. **Target position** (`row, col`)  
3. **Special move flag (optional, fifth element)**  

### **Special Move Encoding**  

| Move Type         | Value |
|------------------|------:|
| Normal move     | 0     |
| King-side castling | 1     |
| Queen-side castling | 2     |
| Queen promotion | 3     |
| Rook promotion  | 4     |
| Bishop promotion | 5     |
| Knight promotion | 6     |

---

## Reward Representation  

- **Games terminate when a player makes an invalid move.**  
- **Reward system:**  
  - **Invalid move:** `-1` penalty.  
  - **Invalid move while in check:** Opponent receives `+1`.  
  - **50-move rule (no captures/promotions in 50 turns):** Draw (`+0.5` for both players).  

> **Note:** Checkmate and stalemate are not explicitly checked to maintain speed. When a player checkmate the other, the game naturally ends the next turn because the losing player has no valid move to make.

---

## Threefold Repetition Rule  

- In standard chess, a player can claim a draw if the same position occurs three times.  
- **In this engine**, instead of tracking full game history, a draw is triggered when **actions are repeated three times**, preventing endless loops in RL training.  

---

## Support the Project  

If you find this useful, consider **starring the repository** ⭐. This helps gauge interest and encourages further development!  

