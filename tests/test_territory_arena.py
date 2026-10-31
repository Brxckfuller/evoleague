import unittest

from evoleague.envs.territory_arena import TerritoryArena


class TestTerritoryArena(unittest.TestCase):

    def test_initial_state(self):
        game = TerritoryArena(size=7)

        self.assertEqual(game.positions, [(0, 0), (6, 6)])
        self.assertEqual(game.scores(), [1, 1])
        self.assertEqual(game.current_player, 0)
        self.assertFalse(game.done)

    def test_movement(self):
        game = TerritoryArena(size=7)

        game.step(3)  # Player 0 moves right

        self.assertEqual(game.positions[0], (0, 1))
        self.assertEqual(game.territory[0][1], 0)

    def test_boundary(self):
        game = TerritoryArena(size=7)

        game.step(0)  # Player 0 attempts to move above top edge

        self.assertEqual(game.positions[0], (0, 0))

    def test_turn_alternation(self):
        game = TerritoryArena(size=7)

        game.step(3)
        self.assertEqual(game.current_player, 1)

        game.step(2)
        self.assertEqual(game.current_player, 0)

    def test_invalid_action(self):
        game = TerritoryArena(size=7)

        with self.assertRaises(ValueError):
            game.step(99)

    def test_reset(self):
        game = TerritoryArena(size=7)

        game.step(3)
        game.reset()

        self.assertEqual(game.positions, [(0, 0), (6, 6)])
        self.assertEqual(game.scores(), [1, 1])
        self.assertEqual(game.turns, 0)
        self.assertFalse(game.done)

    def test_terminal_state(self):
        game = TerritoryArena(size=7, max_turns=2)

        game.step(3)
        _, _, done, _ = game.step(2)

        self.assertTrue(done)

        with self.assertRaises(RuntimeError):
            game.step(3)

    def test_draw(self):
        game = TerritoryArena(size=7, max_turns=2)

        game.step(4)
        _, reward, done, info = game.step(4)

        self.assertTrue(done)
        self.assertIsNone(info["winner"])
        self.assertEqual(reward, 0.0)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            TerritoryArena(size=2)

        with self.assertRaises(ValueError):
            TerritoryArena(max_turns=0)

    def test_observation_immutable_structure(self):
        game = TerritoryArena(size=7)
        observation = game.observe()

        self.assertIsInstance(observation["positions"], tuple)
        self.assertIsInstance(observation["territory"], tuple)


if __name__ == "__main__":
    unittest.main()