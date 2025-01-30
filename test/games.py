import unittest
import chess
import pysrc
import torch

# games chosen from tcec-chess.com

class Suite(unittest.TestCase): 

    def rungame(self, moves):

        chsboard = chess.Board()
        tchboard,tchplayer = pysrc.utils.chessboard2tensor(chsboard)

        # play game on both boards
        for i,move in enumerate(moves):
            #print()
            #print(tchboard[0,:64].view(8,8))
            #print()

            # get actions in san and pwn format
            san = str(chsboard.parse_san(move))
            pwn = pysrc.utils.fen_action2pawner_action(san, chsboard, chsboard.turn)

            # advance chess and pawner boards
            rew,_ = pysrc.step(tchboard,pwn.unsqueeze(0),tchplayer)
            tchplayer = 1 - tchplayer # switch player
            chsboard.push_san(move)

            #if not torch.equal(pysrc.utils.chessboard2tensor(chsboard)[0][:,:64], tchboard[:,:64]) or rew[0,0] != 0:
            #    print("="*10,i,"="*10)
            #    print(san, pwn)
            #    print(pysrc.utils.chessboard2tensor(chsboard)[0][:,:64].view(8,8))
            #    print(tchboard[:,:64].view(8,8))
            #    print(rew)
            #    print(chsboard.unicode())


            # compare chess and pawner boards
            self.assertTrue(torch.equal(pysrc.utils.chessboard2tensor(chsboard)[0][:,:64], tchboard[:,:64]))
            self.assertTrue(rew[0,0] == 0)
            self.assertTrue(rew[0,1] == 0)

        # check no other action is possible
        for pwn_action in pysrc.utils.pawner_actions():
            #print(pwn_action)
            #print(chsboard.unicode())
            pwn = torch.tensor([[*pwn_action]], dtype=torch.int, device="cuda:0")
            rew, _ = pysrc.step(tchboard.clone(),pwn,tchplayer)
            
            self.assertTrue(rew[0,tchplayer[0].item()].item() == -1)
            
    def test_1(self):
        moves = "1. e4 e5 2. f4 exf4 3. Nf3 g5 4. d4 d6 5. g3 g4 6. Nh4 f3 7. Bf4 Nc6 8. Nc3 Bg7 9. Bb5 Kf8 10. Be3 Nce7 11. h3 c6 12. Bd3 h5 13. Qd2 Be6 14. O-O-O Qa5 15. Rde1 Rc8 16. Kb1 c5 17. d5 Bd7 18. e5 c4 19. Be4 Bxe5 20. Bf4 Bf6 21. hxg4 hxg4 22. Bxd6 Bxc3 23. Bxe7+ Nxe7 24. bxc3 Qb6+ 25. Ka1 Qf6 26. Rhf1 Qd6 27. Nxf3 gxf3 28. Rxf3 Rh6 29. Ref1 f5 30. g4 Rc5 31. gxf5 Qf6 32. d6 Qxd6 33. Qc1 Qf6 34. Bxb7 Rb5 35. Be4 Ra5 36. Bb7 Rh2 37. Re3 Ke8 38. Be4 Kd8 39. Qb1 Rb5 40. Qc1 Re2 41. Rd1 Kc8 42. a3 Rf2 43. Rd4 Nxf5 44. Rxc4+ Kb8 45. Bxf5 Qxf5 46. Rd3 a5 47. Rcd4 Be6 48. c4 Rb6 49. Qe3 Rf1+ 50. Rd1 Qxc2 51. Qe5+ Kb7 52. Rd7+ Bxd7 53. Rxf1 Qb3 54. Qd5+ Bc6 55. Qf7+ Ka6 56. Qf2 Qxa3+ 57. Qa2 Qc3+ 58. Qb2 Qb2".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_2(self):
        moves = "1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. Nf3 Bg4 5. Be2 e6 6. O-O Be7 7. h3 Bh5 8. c4 Nb6 9. Nc3 O-O 10. Be3 d5 11. c5 Bxf3 12. gxf3 N6d7 13. f4 Nc6 14. b4 a6 15. Kh2 Kh8 16. Rg1 Rg8 17. Rb1 Nf8 18. a4 Qd7 19. b5 Na5 20. bxa6 bxa6 21. Qe1 Nc6 22. Qf1 Ng6 23. Bxa6 Nh4 24. Qe2 Na5 25. Bd3 g6 26. Nb5 Nc4 27. Ra1 Qc6 28. Rg3 Nf5 29. Rgg1 Nh4 30. Bc1 Rgb8 31. Ra2 Ra5 32. Be3 Raa8 33. Kh1 Nf5 34. Raa1 Nh4 35. Rgc1 Rf8 36. Bd2 Nb2 37. Bc2 Nc4 38. Bb1 Qb7 39. Bc3 f6 40. exf6 Rxf6 41. Bd3 Raf8 42. Re1 Kg8 43. Rad1 e5 44. Bxc4 dxc4+ 45. d5 Bxc5 46. Bxe5 R6f7 47. Qxc4 Bxf2 48. Rf1 Bb6 49. Rd3 Nf5 50. Qc6 Qc8 51. Kh2 Re8 52. Re1 Rfe7 53. Re2 Rf8 54. Qc3 Qe8 55. Qb4 Qa8 56. Qb3 Ref7 57. d6 cxd6 58. Nxd6 Nxd6 59. Bxd6 Re8 60. Rxe8+ Qxe8 61. Be5 Qc8 62. Rd6 Ba5 63. Rd5 Bd8 64. a5 Kf8 65. a6 Ra7 66. Qb8 Qxb8 67. Bxb8 Rxa6 68. Rxd8+ Ke7 69. Rh8 Ke6 70. Rxh7 Ra2+ 71. Kg3 Kf5 72. Rf7+ Ke6 73. Ra7 Rb2 74. f5+ Kf6 75. Ra6+ Kg5 76. Rxg6+ Kxf5 77. Rg8 Rb3+ 78. Kh4 Rb7 79. Rg5+ Ke4 80. Be5 Kf3 81. Bg7 Rb4+ 82. Kh5 Rb6 83. h4 Rb4 84. Re5 Rg4 85. Rf5+ Kg3 86. Rg5 Kf4 87. Rxg4+ Ke3 88. Kg5 Kf3 89. h5 Ke3 90. h6 Kf3 91. h7 Ke3 92. h8=Q Kf3 93. Qh2 Ke3 94. Qe2+ Kxe2 95. Kf4 Kf2 96. Bc3 Ke2 97. Ke4 Kf2 98. Bd4+ Ke2 99. Be3 Kd1 100. Kd3 Ke1 101. Rg1".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_3(self):
        moves = "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 c5 5. Ne2 cxd4 6. exd4 O-O 7. a3 Be7 8. Nf4 d6 9. Be2 Re8 10. O-O Bf8 11. Be3 g6 12. Rc1 Bg7 13. h3 Nc6 14. d5 exd5 15. Nfxd5 Nxd5 16. cxd5 Rxe3 17. fxe3 Ne5 18. Kh1 h5 19. Nb5 Bd7 20. Nd4 Bh6 21. Nf3 Nxf3 22. Rxf3 a6 23. a4 Rb8 24. b4 Qe7 25. Rc7 Qd8 26. Qc1 Bg7 27. b5 axb5 28. Bxb5 Rc8 29. Rxc8 Bxc8 30. Bd3 Be5 31. a5 Bd7 32. Qe1 Ba4 33. Rf1 Kg7 34. Qb4 Be8 35. Rb1 Bd7 36. Qd2 Qg5 37. Qf2 Bc8 38. Rf1 Qe7 39. Rc1 Qd8 40. Kg1 Bd7 41. Qe1 Be8 42. Rb1 Qc7 43. Qb4 Qd8 44. Kf1 Qg5 45. Qd2 Qe7 46. Qc2 Kh6 47. Qc8 Bg3 48. Ke2 Bf4 49. Qc1 Be5 50. Rb6 Qg5 51. Be4 Qh4 52. Qc2 Qd8 53. Qc4 Bg3 54. Bf3 Qd7 55. Qb4 Qf5 56. Qc3 Be5 57. Qc4 Qd7 58. Kd2 Kh7 59. Qb4 Qc8 60. g4 Bd7 61. gxh5 gxh5 62. Rxb7 Bc3+ 63. Qxc3 Qxb7 64. Qa3 Bc8 65. Bxh5 Kg8 66. Qxd6 Qb5 67. Qd8+ Kh7 68. Qxc8 Qxd5+ 69. Kc1 Qxh5 70. Qc4 Kh6 71. Qb4 Qd5 72. Qb6+ Kh7 73. a6 Qc4+ 74. Kb2 Qe2+ 75. Kc3 Qe1+ 76. Kd4 Qa1+ 77. Kd5 Qa2+ 78. Kc5 Qf2 79. Qd6 Qa2 80. Kb6 Qb3+ 81. Kc7 Qc3+ 82. Kd7 Kg8 83. a7 Qa5 84. Qb8+ Kg7 85. Qb2+ Kh7 86. Qb7 Qf5+ 87. Ke8 Kg6 88. a8=Q Qe5+ 89. Qe7 Qh8+ 90. Qf8 Qe5+ 91. Kd8 Qf6+ 92. Kc7 Qc3+ 93. Qc6+ Qxc6+ 94. Kxc6 Kf5 95. Kd5 Kg6 96. Ke5 Kh7 97. Qxf7+ Kh8 98. Qb7 Kg8 99. Kf6 Kh8 100. Qg7#".split(" ")
        del moves[::3]
        self.rungame(moves)
