import unittest 
from test.kingside_castle import Suite as KingsideCastleSuite
from test.queenside_castle import Suite as QueensideCastleSuite
from test.promotion import Suite as PromotionSuite
from test.attacks import Suite as AttacksSuite
from test.pawn_move import Suite as PawnMoveSuite
from test.pawn_double import Suite as PawnDoubleSuite

if __name__ == "__main__":
    # Create a TestLoader instance
    loader = unittest.TestLoader()

    # Load tests from the test classes
    attacks_suite = loader.loadTestsFromTestCase(AttacksSuite)
    kingside_suite = loader.loadTestsFromTestCase(KingsideCastleSuite)
    queenside_suite = loader.loadTestsFromTestCase(QueensideCastleSuite)
    promotion_suite = loader.loadTestsFromTestCase(PromotionSuite)
    pawn_move_suite = loader.loadTestsFromTestCase(PawnMoveSuite)
    pawn_double_suite = loader.loadTestsFromTestCase(PawnDoubleSuite)

    # Combine the tests into a single suite
    test_suite = unittest.TestSuite([
        pawn_double_suite,
        pawn_move_suite,
        attacks_suite, 
        kingside_suite, 
        queenside_suite, 
        promotion_suite
    ])

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
