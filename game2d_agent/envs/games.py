"""
Game presets.

A "game" is a set of world parameters over the same engine (PixelGameEnv). The
agent's perception is egocentric/relative, so it should transfer across these
without observation-shape changes — that is exactly what gate 3 measures.

Keep in sync with docs/game_mechanics.md.
"""

GAMES = {
    # Game A — the game everything was developed on.
    "collect2d": dict(
        screen_size=(640, 480),
        obstacle_size=(60, 120),
    ),
    # Game B — transfer target (gate 3): bigger arena, wider obstacle-size range,
    # same "collect all targets" objective.
    "collect2d_b": dict(
        screen_size=(800, 600),
        obstacle_size=(50, 160),
    ),
}


def game_kwargs(name: str) -> dict:
    if name not in GAMES:
        raise ValueError(f"unknown game '{name}'. Known: {list(GAMES)}")
    return dict(GAMES[name])
