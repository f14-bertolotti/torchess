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
        fen_actions = []
        pwn_actions = []
        for pwn_action in utils.pwn_actions():
            try:
                fen_action = utils.pwn2san(pwn_action, chess.WHITE)
            except ValueError: continue
        
            chess_board = utils.str2chs(stringboard, turn, rights)
            torch_board = utils.chs2pwn(chess_board)

            torch_err = pysrc.step(torch_board[0], torch.tensor([pwn_action], dtype=torch.int, device="cuda:0"), torch_board[1])[0]
            torch_err = -int(torch_err[0,0 if turn == chess.WHITE else 1].item())

            try:
                chess.Move.from_uci(fen_action)
                chess_board.push_uci(fen_action)
                chess_err = 0
            except Exception:
                chess_err = 1
        
            self.assertEqual(chess_err, torch_err)
            if chess_err == 0:
                self.assertTrue(torch.equal(utils.chs2pwn(chess_board)[0][0,:64], torch_board[0][0,:64]))

            if chess_err == 0: pwn_actions.append(pwn_action)
            if torch_err == 0: fen_actions.append(fen_action)

        return fen_actions, pwn_actions


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

    def test_winning_white(self):
        stringboard,turn,rights = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ♕ ♚
        ⭘ ⭘ ⭘ ⭘ ⭘ ♟ ⭘ ♙
        ⭘ ⭘ ⭘ ⭘ ♙ ♛ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ♙ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ♙ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ♔ ⭘ ⭘ ♖ ⭘
        """, chess.BLACK, ""

        fen_actions, pwn_actions = self.trymoves(stringboard,turn,rights)
        self.assertEqual(fen_actions, [])
        self.assertEqual(pwn_actions, [])

    def test_winning_black(self):
        stringboard,turn,rights = """
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♚ ♜ ♝ ⭘ ⭘ ⭘ ⭘ ⭘
        ♟ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ♙ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ⭘ ♛ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
        ♔ ⭘ ⭘ ⭘ ⭘ ♖ ⭘ ⭘
        """, chess.WHITE, ""

        fen_actions, pwn_actions = self.trymoves(stringboard,turn,rights)
        self.assertEqual(fen_actions, [])
        self.assertEqual(pwn_actions, [])
