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
            #print(i,end=" ")
            #print()
            #print(tchboard[0,:64].view(8,8))
            #print(chsboard.unicode())
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
            #print(rew)
            
            self.assertTrue(rew[0,tchplayer[0].item()].item() == -1)
            
    def test_1(self):
        moves = "1. e4 e5 2. f4 exf4 3. Nf3 g5 4. d4 d6 5. g3 g4 6. Nh4 f3 7. Bf4 Nc6 8. Nc3 Bg7 9. Bb5 Kf8 10. Be3 Nce7 11. h3 c6 12. Bd3 h5 13. Qd2 Be6 14. O-O-O Qa5 15. Rde1 Rc8 16. Kb1 c5 17. d5 Bd7 18. e5 c4 19. Be4 Bxe5 20. Bf4 Bf6 21. hxg4 hxg4 22. Bxd6 Bxc3 23. Bxe7+ Nxe7 24. bxc3 Qb6+ 25. Ka1 Qf6 26. Rhf1 Qd6 27. Nxf3 gxf3 28. Rxf3 Rh6 29. Ref1 f5 30. g4 Rc5 31. gxf5 Qf6 32. d6 Qxd6 33. Qc1 Qf6 34. Bxb7 Rb5 35. Be4 Ra5 36. Bb7 Rh2 37. Re3 Ke8 38. Be4 Kd8 39. Qb1 Rb5 40. Qc1 Re2 41. Rd1 Kc8 42. a3 Rf2 43. Rd4 Nxf5 44. Rxc4+ Kb8 45. Bxf5 Qxf5 46. Rd3 a5 47. Rcd4 Be6 48. c4 Rb6 49. Qe3 Rf1+ 50. Rd1 Qxc2 51. Qe5+ Kb7 52. Rd7+ Bxd7 53. Rxf1 Qb3 54. Qd5+ Bc6 55. Qf7+ Ka6 56. Qf2 Qxa3+ 57. Qa2 Qc3+ 58. Qb2 Qb2#".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_2(self):
        moves = "1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. Nf3 Bg4 5. Be2 e6 6. O-O Be7 7. h3 Bh5 8. c4 Nb6 9. Nc3 O-O 10. Be3 d5 11. c5 Bxf3 12. gxf3 N6d7 13. f4 Nc6 14. b4 a6 15. Kh2 Kh8 16. Rg1 Rg8 17. Rb1 Nf8 18. a4 Qd7 19. b5 Na5 20. bxa6 bxa6 21. Qe1 Nc6 22. Qf1 Ng6 23. Bxa6 Nh4 24. Qe2 Na5 25. Bd3 g6 26. Nb5 Nc4 27. Ra1 Qc6 28. Rg3 Nf5 29. Rgg1 Nh4 30. Bc1 Rgb8 31. Ra2 Ra5 32. Be3 Raa8 33. Kh1 Nf5 34. Raa1 Nh4 35. Rgc1 Rf8 36. Bd2 Nb2 37. Bc2 Nc4 38. Bb1 Qb7 39. Bc3 f6 40. exf6 Rxf6 41. Bd3 Raf8 42. Re1 Kg8 43. Rad1 e5 44. Bxc4 dxc4+ 45. d5 Bxc5 46. Bxe5 R6f7 47. Qxc4 Bxf2 48. Rf1 Bb6 49. Rd3 Nf5 50. Qc6 Qc8 51. Kh2 Re8 52. Re1 Rfe7 53. Re2 Rf8 54. Qc3 Qe8 55. Qb4 Qa8 56. Qb3 Ref7 57. d6 cxd6 58. Nxd6 Nxd6 59. Bxd6 Re8 60. Rxe8+ Qxe8 61. Be5 Qc8 62. Rd6 Ba5 63. Rd5 Bd8 64. a5 Kf8 65. a6 Ra7 66. Qb8 Qxb8 67. Bxb8 Rxa6 68. Rxd8+ Ke7 69. Rh8 Ke6 70. Rxh7 Ra2+ 71. Kg3 Kf5 72. Rf7+ Ke6 73. Ra7 Rb2 74. f5+ Kf6 75. Ra6+ Kg5 76. Rxg6+ Kxf5 77. Rg8 Rb3+ 78. Kh4 Rb7 79. Rg5+ Ke4 80. Be5 Kf3 81. Bg7 Rb4+ 82. Kh5 Rb6 83. h4 Rb4 84. Re5 Rg4 85. Rf5+ Kg3 86. Rg5 Kf4 87. Rxg4+ Ke3 88. Kg5 Kf3 89. h5 Ke3 90. h6 Kf3 91. h7 Ke3 92. h8=Q Kf3 93. Qh2 Ke3 94. Qe2+ Kxe2 95. Kf4 Kf2 96. Bc3 Ke2 97. Ke4 Kf2 98. Bd4+ Ke2 99. Be3 Kd1 100. Kd3 Ke1 101. Rg1#".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_3(self):
        moves = "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 c5 5. Ne2 cxd4 6. exd4 O-O 7. a3 Be7 8. Nf4 d6 9. Be2 Re8 10. O-O Bf8 11. Be3 g6 12. Rc1 Bg7 13. h3 Nc6 14. d5 exd5 15. Nfxd5 Nxd5 16. cxd5 Rxe3 17. fxe3 Ne5 18. Kh1 h5 19. Nb5 Bd7 20. Nd4 Bh6 21. Nf3 Nxf3 22. Rxf3 a6 23. a4 Rb8 24. b4 Qe7 25. Rc7 Qd8 26. Qc1 Bg7 27. b5 axb5 28. Bxb5 Rc8 29. Rxc8 Bxc8 30. Bd3 Be5 31. a5 Bd7 32. Qe1 Ba4 33. Rf1 Kg7 34. Qb4 Be8 35. Rb1 Bd7 36. Qd2 Qg5 37. Qf2 Bc8 38. Rf1 Qe7 39. Rc1 Qd8 40. Kg1 Bd7 41. Qe1 Be8 42. Rb1 Qc7 43. Qb4 Qd8 44. Kf1 Qg5 45. Qd2 Qe7 46. Qc2 Kh6 47. Qc8 Bg3 48. Ke2 Bf4 49. Qc1 Be5 50. Rb6 Qg5 51. Be4 Qh4 52. Qc2 Qd8 53. Qc4 Bg3 54. Bf3 Qd7 55. Qb4 Qf5 56. Qc3 Be5 57. Qc4 Qd7 58. Kd2 Kh7 59. Qb4 Qc8 60. g4 Bd7 61. gxh5 gxh5 62. Rxb7 Bc3+ 63. Qxc3 Qxb7 64. Qa3 Bc8 65. Bxh5 Kg8 66. Qxd6 Qb5 67. Qd8+ Kh7 68. Qxc8 Qxd5+ 69. Kc1 Qxh5 70. Qc4 Kh6 71. Qb4 Qd5 72. Qb6+ Kh7 73. a6 Qc4+ 74. Kb2 Qe2+ 75. Kc3 Qe1+ 76. Kd4 Qa1+ 77. Kd5 Qa2+ 78. Kc5 Qf2 79. Qd6 Qa2 80. Kb6 Qb3+ 81. Kc7 Qc3+ 82. Kd7 Kg8 83. a7 Qa5 84. Qb8+ Kg7 85. Qb2+ Kh7 86. Qb7 Qf5+ 87. Ke8 Kg6 88. a8=Q Qe5+ 89. Qe7 Qh8+ 90. Qf8 Qe5+ 91. Kd8 Qf6+ 92. Kc7 Qc3+ 93. Qc6+ Qxc6+ 94. Kxc6 Kf5 95. Kd5 Kg6 96. Ke5 Kh7 97. Qxf7+ Kh8 98. Qb7 Kg8 99. Kf6 Kh8 100. Qg7#".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_4(self):
        moves = "1. d4 Nf6 2. Nf3 e6 3. Bg5 b6 4. e4 h6 5. Bxf6 Qxf6 6. Bd3 Bb7 7. Nbd2 d6 8. Qe2 a6 9. O-O-O Nd7 10. Kb1 e5 11. c3 Be7 12. h4 c5 13. Nc4 b5 14. Ne3 Nb6 15. Rhe1 O-O-O 16. dxe5 dxe5 17. c4 Rd6 18. Nd5 Nxd5 19. exd5 bxc4 20. Bxc4 Rhd8 21. Qe4 Qe6 22. Re3 f6 23. h5 Qd7 24. Red3 Kb8 25. Nh4 Bc8 26. Ng6 Rb6 27. Nxe7 Qxe7 28. Rc3 Rd7 29. Ka1 Rdb7 30. Rd2 Qd6 31. a3 Rc7 32. Ba2 Rbb7 33. Qg6 Re7 34. Rdc2 Rbc7 35. f3 Bd7 36. Rc1 Be8 37. Qg4 Ka8 38. Bb1 Bd7 39. Qg6 Bb5 40. Qe4 Re8 41. Qe3 c4 42. Ba2 Rec8 43. Qe4 f5 44. Qxf5 Qxd5 45. Qc2 Ka7 46. Rd1 Qc6 47. Qe2 Qc5 48. Qe1 Kb8 49. Re3 e4 50. fxe4 Qxh5 51. e5 Qg5 52. e6 Qf6 53. Rc1 Qe7 54. Qc3 Be8 55. Qe5 Bb5 56. Bxc4 Qf6 57. Qxf6 gxf6 58. b3 Re7 59. Kb2 Rc5 60. Be2 Re5 61. Rxe5 fxe5 62. Bg4 Rg7 63. Bh3 Rc7 64. Re1 Bd3 65. b4 e4 66. Bf5 Rc2+ 67. Kb3 Rc7 68. g4 a5 69. bxa5 Ka7 70. a4 Bc4+ 71. Ka3 Re7 72. Rxe4 Ba6 73. Kb4 Kb7 74. Kc5 h5 75. Kd6 Re8 76. e7 Ka7 77. Bd7 Rg8 78. Rf4 Bc8 79. Bc6 Rg6+ 80. Kc5 Rg5+ 81. Kd4 Rg8 82. g5 Bb7 83. Bb5 Rc8 84. Rf8 Rc1 85. e8=Q Rc8 86. Qxc8 Bxc8 87. Rxc8 Kb7 88. Ba6+ Kxa6 89. g6 Kb7 90. Rc5 h4 91. g7 h3 92. g8=Q h2 93. Qc8+ Ka7 94. Rc7#".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_5(self):
        moves = "1. c4 e6 2. d4 d5 3. Nf3 c6 4. Nc3 f5 5. Bf4 Nf6 6. e3 Be7 7. Be2 O-O 8. O-O Nh5 9. g3 Nd7 10. h4 Nhf6 11. Ng5 Nb8 12. Kg2 h6 13. Rh1 Ne4 14. Ncxe4 fxe4 15. Bh5 hxg5 16. hxg5 Bxg5 17. Bg6 Rxf4 18. gxf4 Bh6 19. Rxh6 gxh6 20. Qh5 Qf6 21. Rg1 Kf8 22. Kf1 Nd7 23. Bh7 Ke7 24. Rg6 Qf8 25. Qh4+ Nf6 26. Rxh6 b6 27. Qg5 Ba6 28. Ke1 Bxc4 29. f5 Ba6 30. a3 c5 31. fxe6 Bc8 32. Bf5 Bxe6 33. Rh7+ Ke8 34. Bg6+ Kd8 35. Bf7 Qe7 36. Rh8+ Kc7 37. Rxa8 Kb7 38. Bxe6 Kxa8 39. Bxd5+ Nxd5 40. Qxd5+ Kb8 41. dxc5 bxc5 42. a4 Qe8 43. Qd6+ Kb7 44. b3 Qh8 45. Qd7+ Ka8 46. Qc6+ Kb8 47. Qb5+ Ka8 48. Qxc5 Qh1+ 49. Kd2 Kb8 50. Qd4 Qb1 51. a5 Qa2+ 52. Kc3 Qxf2 53. a6 Qh4 54. Qd6+ Kc8 55. Qc6+ Kd8 56. Qa8+ Kc7 57. Qxa7+ Kd8 58. Qd4+ Kc8 59. Qc5+ Kb8 60. Qb6+ Kc8 61. Qc6+ Kd8 62. a7 Qe1+ 63. Kc4 Qe2+ 64. Kb4 Qa2 65. a8=Q+ Qxa8 66. Qxa8+ Ke7 67. Kc5 Ke6 68. Qg8+ Ke5 69. Qf8 Ke6 70. Kc6 Ke5 71. Kd7 Kd5 72. Qf5#".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_6(self):
        moves = "1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 Nf5 8. Bd3 h5 9. Qh3 Nc6 10. Nf3 Qa5 11. O-O cxd4 12. cxd4 Nfxd4 13. Nxd4 Nxd4 14. Rb1 Qc7 15. Bf4 Bd7 16. Rfc1 Kf8 17. Qe3 Nc6 18. c4 d4 19. Qe4 h4 20. h3 Be8 21. Rc2 Rc8 22. c5 Rh5 23. Be2 Rf5 24. Bh2 Qd8 25. Rcb2 Ne7 26. Qxh4 Bc6 27. Bd3 Kg8 28. f3 Qd7 29. Re1 Qc7 30. Rd1 Ba4 31. Rc1 Bc6 32. Qxd4 Rd8 33. Qc3 Rh5 34. Rd2 Rd5 35. Bg3 Rg5 36. Kh2 Rxg3 37. Kxg3 Rxe5 38. Kf2 Rg5 39. Qd4 Rd5 40. Qb4 a5 41. Qb2 Qf4 42. Rc4 Qg5 43. Rg4 Qh6 44. Rd1 f5 45. Rd4 Rxc5 46. Rd8+ Kf7 47. Qd4 b6 48. Re1 Bd5 49. Qb2 Rc7 50. Qxb6 Rb7 51. Qc5 Qh4+ 52. Kf1 Qf4 53. Be2 a4 54. Ra8 Bb3 55. Ra7 Rxa7 56. Qxa7 Qd6 57. Bb5 Qxa3 58. Qd4 Bd5 59. Ra1 Qb3 60. Bxa4 Qb8 61. Bc2 Nc6 62. Qc3 g5 63. Rb1 Qa7 64. Bd3 Ne7 65. Be2 Ba2 66. Rb4 Kg6 67. h4 Nd5 68. Qh8 Ne3+ 69. Kg1 g4 70. Kh2 Qg7 71. h5+ Kf6 72. Qd8+ Qe7 73. Qxe7+ Kxe7 74. Rb7+ Kf8 75. Kg3 Bd5 76. Rc7 Kg8 77. h6 Ba2 78. Kh4 Nd5 79. Ra7 Bb1 80. Kg5 f4 81. fxg4 Ne3 82. Kf6 Nd5+ 83. Ke5 Nb6 84. g5 f3 85. Bxf3 Nc8 86. Rb7 Bd3 87. Bh5 Be4 88. Rb8 Kh7 89. Rxc8 Bb7 90. Rc7+ Kh8 91. Rxb7 Kg8 92. g6 Kf8 93. Kxe6 Kg8 94. Rb8#".split(" ")
        del moves[::3]
        self.rungame(moves)

    def test_7(self):
        moves = "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Be2 O-O 6. Nf3 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Ne8 10. Be3 f5 11. f3 Kh8 12. a4 Ng8 13. a5 Bh6 14. Bf2 Ngf6 15. b4 Qe7 16. Nd3 Nh5 17. exf5 gxf5 18. g3 Bg7 19. Ra3 Qg5 20. Kh1 Nef6 21. Rg1 a6 22. Nb2 Qh6 23. Be1 Nd7 24. Rg2 Re8 25. Ra2 Qg6 26. c5 dxc5 27. bxc5 Nhf6 28. Nc4 Nxc5 29. Bf2 Ncd7 30. Na4 Qf7 31. Rd2 Qf8 32. d6 b5 33. axb6 cxb6 34. Naxb6 Rb8 35. Nxc8 Rexc8 36. Ba7 Rb5 37. Bg1 Rcb8 38. Qa4 Nc5 39. Qa2 Nfd7 40. Rd1 Qf7 41. Bf1 Rb4 42. Rgd2 R4b7 43. Rc2 f4 44. Nd2 Qxa2 45. Rxa2 Bf8 46. gxf4 exf4 47. Bd4+ Bg7 48. Bxc5 Nxc5 49. Rc2 Nd7 50. Ne4 Bf6 51. Nxf6 Nxf6 52. Bh3 Rb1 53. Rxb1 Rxb1+ 54. Kg2 Rd1 55. Rc8+ Kg7 56. d7 Nxd7 57. Rc7 Rd2+ 58. Kf1 Rxh2 59. Bxd7 Kg6 60. Rc6+ Kg5 61. Rxa6 Rb2 62. Bc6 Rb1+ 63. Kg2 Rc1 64. Be4 Rc4 65. Bxh7 Rc5 66. Be4 Re5 67. Kh3 Kh5 68. Rf6 Kg5 69. Rc6 Kh5 70. Rd6 Ra5 71. Rd5+ Rxd5 72. Bxd5 Kg5 73. Bc6 Kh5 74. Be4 Kg5 75. Bd3 Kh5 76. Bf1 Kg5 77. Ba6 Kh5 78. Bc4 Kg5 79. Bd5 Kh5 80. Bb3 Kh6 81. Kh4 Kg7 82. Bc4 Kh7 83. Bf7 Kh8 84. Be6 Kh7 85. Kg5 Kg7 86. Bc4 Kh7 87. Bd3+ Kg7 88. Bb5 Kh7 89. Bd7 Kg7 90. Be6 Kf8 91. Ba2 Kg7 92. Kxf4 Kh6 93. Bd5 Kg6 94. Bb7 Kf6 95. Ba6 Kf7 96. Bb5 Ke6 97. Bc4+ Kd6 98. Ke4 Kc6 99. Ke5 Kb6 100. f4 Kc6 101. Bg8 Kd7 102. Ke4 Kd6 103. Ba2 Kc5 104. Ke5 Kc6 105. f5 Kd7 106. f6 Ke8 107. Bd5 Kd7 108. Bc6+ Kxc6 109. f7 Kb5 110. Kf6 Ka4 111. f8=Q Kb5 112. Qa3 Kc6 113. Qb4 Kd5 114. Kf5 Kc6 115. Ke6 Kc7 116. Qb5 Kc8 117. Kd6 Kd8 118. Qd7#".split(" ")
        del moves[::3]
        self.rungame(moves)
