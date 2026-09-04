import io
import re
from datetime import datetime
from pathlib import Path

import chess.pgn

DEFAULT_PGN_DIR = Path("/Users/ramiar/Projects/chessathon/game-data")


def save_pgn(
    directory: Path,
    pgn: str,
    white: Path,
    black: Path,
    base_ms: int,
    increment_ms: int,
    game_number: int = 1,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    played_at = datetime.now().astimezone()
    white_name = _label(white)
    black_name = _label(black)
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("could not parse generated PGN")
    game.headers.update(
        {
            "Event": "AI Chessathon local benchmark",
            "Site": "local",
            "Date": played_at.strftime("%Y.%m.%d"),
            "White": white_name,
            "Black": black_name,
            "Time": played_at.strftime("%H:%M:%S%z"),
            "TimeControl": f"{base_ms / 1000:g}+{increment_ms / 1000:g}",
        }
    )
    timestamp = played_at.strftime("%Y%m%dT%H%M%S%z")
    filename = f"{timestamp}__{white_name}-vs-{black_name}__game-{game_number:03d}.pgn"
    path = directory / filename
    path.write_text(str(game) + "\n")
    return path


def _label(directory: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", directory.name).strip("-") or "agent"
