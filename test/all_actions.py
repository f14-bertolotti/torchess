import pysrc.utils as utils
import unittest
import chess
import torch
import pysrc
import tqdm

reference_board = """
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
"""


class Suite(unittest.TestCase): 

    def trymoves(self,stringboard,turn,rights):
        for pawner_action in (msg:=tqdm.tqdm(utils.pawner_actions(), total=8*8*8*8*7)):
            try:
                fen_action = utils.pawner_action2fen_action(pawner_action, chess.WHITE)
            except ValueError: continue
        
            chess_board = utils.string2chessboard(stringboard, turn, rights)
            torch_board = utils.chessboard2tensor(chess_board)
            torch_err = pysrc.step(torch_board[0], torch.tensor([pawner_action], dtype=torch.int, device="cuda:0"), torch_board[1])[0]
        
            try:
                chess.Move.from_uci(fen_action)
                chess_board.push_uci(fen_action)
                chess_err = 0
            except Exception:
                chess_err = 1
        
            msg.set_description(f"{pawner_action} {fen_action: <5}")
        
            self.assertEqual(chess_err, -int(torch_err[0,0 if turn == chess.WHITE else 1].item()))
            if chess_err == 0:
                self.assertEqual(utils.chessboard2tensor(chess_board)[0].tolist(), torch_board[0].tolist())



    def test_intial_board_white(self):
        stringboard,turn,rights = """
        ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
        """, chess.WHITE, "KQkq"

        self.trymoves(stringboard,turn,rights)



    def test_intial_board_black(self):
        stringboard,turn,rights = """
        ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
        ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
        """, chess.BLACK, "KQkq"

        self.trymoves(stringboard,turn,rights)
