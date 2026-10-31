from dataclasses import dataclass, field


@dataclass
class TerritoryArena:
    size: int = 7
    max_turns: int = 100

    positions: list = field(default_factory=list)
    territory: list = field(default_factory=list)
    current_player: int = 0
    turns: int = 0
    done: bool = False

    ACTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
        4: (0, 0),   # Stay
    }

    def __post_init__(self):
        if self.size < 3:
            raise ValueError("Arena size must be at least 3.")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive.")
        self.reset()

    def reset(self):
        self.positions = [
            (0, 0),
            (self.size - 1, self.size - 1),
        ]

        self.territory = [
            [-1 for _ in range(self.size)]
            for _ in range(self.size)
        ]

        for player, (row, col) in enumerate(self.positions):
            self.territory[row][col] = player

        self.current_player = 0
        self.turns = 0
        self.done = False

        return self.observe()

    def observe(self):
        return {
            "positions": tuple(self.positions),
            "territory": tuple(
                tuple(row) for row in self.territory
            ),
            "current_player": self.current_player,
            "turns_remaining": self.max_turns - self.turns,
        }

    def encode_observation(self, player):
        if player not in (0, 1):
            raise ValueError("Player must be 0 or 1.")

        opponent = 1 - player

        territory_features = []
        for row in self.territory:
            for owner in row:
                if owner == -1:
                    territory_features.append(0.0)
                elif owner == player:
                    territory_features.append(1.0)
                else:
                    territory_features.append(-1.0)

        def position_features(position):
            features = [0.0] * (self.size * self.size)
            row, col = position
            features[row * self.size + col] = 1.0
            return features

        own_position = position_features(self.positions[player])
        opponent_position = position_features(self.positions[opponent])

        remaining_turns = [
            (self.max_turns - self.turns) / self.max_turns
        ]

        return (
                territory_features
                + own_position
                + opponent_position
                + remaining_turns
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Game has already ended.")

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        player = self.current_player
        row, col = self.positions[player]

        dr, dc = self.ACTIONS[action]

        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))

        self.positions[player] = (new_row, new_col)
        self.territory[new_row][new_col] = player

        self.turns += 1

        rewards = (0.0, 0.0)
        winner = None

        if self.turns >= self.max_turns:
            self.done = True
            scores = self.scores()

            if scores[0] > scores[1]:
                winner = 0
                rewards = (1.0, -1.0)
            elif scores[1] > scores[0]:
                winner = 1
                rewards = (-1.0, 1.0)
        else:
            self.current_player = 1 - player

        return self.observe(), rewards, self.done, {
            "winner": winner
        }

    def scores(self):
        return [
            sum(cell == player for row in self.territory for cell in row)
            for player in (0, 1)
        ]

    def render(self):
        symbols = {
            -1: ".",
            0: "B",
            1: "R",
        }

        grid = [
            [symbols[cell] for cell in row]
            for row in self.territory
        ]

        for player, (row, col) in enumerate(self.positions):
            grid[row][col] = "0" if player == 0 else "1"

        print()
        for row in grid:
            print(" ".join(row))

        print(
            f"\nTurn: {self.turns}/{self.max_turns}"
            f" | Scores: {self.scores()}"
            f" | Current player: {self.current_player}"
        )

