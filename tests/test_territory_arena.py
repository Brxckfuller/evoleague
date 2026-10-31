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
        _, rewards, done, info = game.step(4)

        self.assertTrue(done)
        self.assertIsNone(info["winner"])
        self.assertEqual(rewards, (0.0, 0.0))

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

    def test_player_zero_wins(self):
        game = TerritoryArena(size=7, max_turns=2)

        game.step(3)
        _, rewards, done, info = game.step(4)

        self.assertTrue(done)
        self.assertEqual(info["winner"], 0)
        self.assertEqual(rewards, (1.0, -1.0))

    def test_player_one_wins(self):
        game = TerritoryArena(size=7, max_turns=2)

        game.step(4)
        _, rewards, done, info = game.step(2)

        self.assertTrue(done)
        self.assertEqual(info["winner"], 1)
        self.assertEqual(rewards, (-1.0, 1.0))

    def test_non_terminal_rewards(self):
        game = TerritoryArena(size=7, max_turns=4)

        _, rewards, done, info = game.step(3)

        self.assertFalse(done)
        self.assertEqual(rewards, (0.0, 0.0))
        self.assertIsNone(info["winner"])

    def test_shared_cell_occupancy(self):
        game = TerritoryArena(size=3, max_turns=10)

        game.step(3)  # Player 0: (0, 0) -> (0, 1)
        game.step(0)  # Player 1: (2, 2) -> (1, 2)
        game.step(3)  # Player 0: (0, 1) -> (0, 2)
        game.step(0)  # Player 1: (1, 2) -> (0, 2)

        self.assertEqual(game.positions[0], (0, 2))
        self.assertEqual(game.positions[1], (0, 2))
        self.assertEqual(game.territory[0][2], 1)

    def test_territory_recapture(self):
        game = TerritoryArena(size=3, max_turns=10)

        game.step(3)  # Player 0 -> (0, 1)
        game.step(0)  # Player 1 -> (1, 2)
        game.step(3)  # Player 0 -> (0, 2)
        game.step(0)  # Player 1 -> (0, 2), captures it

        self.assertEqual(game.territory[0][2], 1)

        game.step(4)  # Player 0 stays at (0, 2), recaptures it

        self.assertEqual(game.territory[0][2], 0)

    def test_encoded_observation_length(self):
        game = TerritoryArena(size=7)

        observation = game.encode_observation(player=0)

        self.assertEqual(len(observation), 148)
        self.assertTrue(all(isinstance(x, float) for x in observation))

    def test_encoded_observation_perspective(self):
        game = TerritoryArena(size=3)

        player_zero = game.encode_observation(player=0)
        player_one = game.encode_observation(player=1)

        # Player 0's starting cell
        self.assertEqual(player_zero[0], 1.0)
        self.assertEqual(player_one[0], -1.0)

        # Player 1's starting cell
        self.assertEqual(player_zero[8], -1.0)
        self.assertEqual(player_one[8], 1.0)

        # Each player's own position is encoded separately
        self.assertEqual(player_zero[9], 1.0)
        self.assertEqual(player_one[17], 1.0)

    def test_encoded_observation_updates(self):
        game = TerritoryArena(size=3, max_turns=10)

        game.step(3)

        observation = game.encode_observation(player=0)

        self.assertEqual(observation[1], 1.0)
        self.assertEqual(observation[-1], 0.9)


if __name__ == "__main__":
    unittest.main()