"""The submission entrypoint. The platform imports this file and calls get_move."""

import math
import random
import time

import chess
import numpy as np
from numba import njit

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int32)
MOBILITY_WEIGHT = 4
MATE = 1e6

# --- Time management tuning -------------------------------------------------
# Time control per the brief: 120s + 0.5s/move. time_left_ms is BEFORE this move's
# increment is added, but the increment is guaranteed regardless of how low the
# clock gets, so we plan on spending most of it every move.

INCREMENT_MS = 500
MOVES_DIVISOR = 30          # crude "assume ~30 moves left" budget split of the base clock
MOVE_OVERHEAD_MS = 50       # safety margin subtracted for interprocess/GC/wire overhead
MIN_BUDGET_MS = 10          # always try for at least this much, even in extreme time trouble
TIME_CHECK_INTERVAL = 1024  # check the clock every N nodes, not every node (perf_counter has cost)
MAX_DEPTH = 64              # hard ceiling; the time budget will stop us long before this

# --- Move ordering tuning ----------------------------------------------------
# Captures and promotions are searched before quiet moves. This is what makes alpha-beta
# pruning actually cut the tree: a cutoff only happens once we've found a move good enough
# to prove the rest of the branch irrelevant, and the biggest, most forcing swings in
# material are the moves most likely to be that good.

CAPTURE_BASE = 10_000
PROMOTION_BASE = 20_000

class TimeUp(Exception):
    """Raised inside the search once this move's deadline has passed."""

@njit(cache=False)
def evaluate(pieces: np.ndarray, mine: np.ndarray, mobility: int) -> int:
    material = 0
    for square in range(64):
        piece = pieces[square]
        if piece == 0:
            continue
        value = PIECE_VALUE[piece]
        material += value if mine[square] else -value
    return material + MOBILITY_WEIGHT * mobility

def encode(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    pieces = np.zeros(64, dtype=np.int32)
    mine = np.zeros(64, dtype=np.bool_)
    for square, piece in board.piece_map().items():
        pieces[square] = piece.piece_type
        mine[square] = piece.color == board.turn
    return pieces, mine

def move_priority(board: chess.Board, move: chess.Move) -> int:
    """Higher sorts first. Captures ranked by MVV-LVA (Most Valuable Victim, Least
    Valuable Attacker) -- capturing a queen with a pawn ranks far above capturing a
    pawn with a queen, even though both are "a capture". Promotions get their own
    boost since they're similarly forcing. Quiet moves score 0 and sort last, in
    whatever order legal_moves produced them (no ordering among them yet)."""
    priority = 0
    if move.promotion:
        priority += PROMOTION_BASE + int(PIECE_VALUE[move.promotion])
    if board.is_capture(move):
        if board.is_en_passant(move):
            victim_type = chess.PAWN
        else:
            victim_type = board.piece_type_at(move.to_square)
        attacker_type = board.piece_type_at(move.from_square)
        priority += CAPTURE_BASE + int(PIECE_VALUE[victim_type]) * 10 - int(PIECE_VALUE[attacker_type])
    return priority

def order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    return sorted(moves, key=lambda move: move_priority(board, move), reverse=True)

class Searcher:
    """Holds per-move search state (node counter + deadline) so negamax can check the
    clock without threading a deadline argument through every recursive call."""
 
    def __init__(self, deadline: float):
        self.deadline = deadline
        self.nodes = 0
 
    def check_time(self) -> None:
        self.nodes += 1
        if self.nodes % TIME_CHECK_INTERVAL == 0 and time.perf_counter() >= self.deadline:
            raise TimeUp
 
 
    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        self.check_time()
        moves = list(board.legal_moves)
        if not moves:
            return -MATE if board.is_check() else 0.0
        if depth == 0:
            pieces, mine = encode(board)
            return float(evaluate(pieces, mine, len(moves)))
 
        value = -math.inf
        for move in order_moves(board, moves):
            board.push(move)
            try:
                value = max(value, -self.negamax(board, depth - 1, -beta, -alpha))
            finally:
                board.pop()
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break  # beta cutoff: the opponent already has a better option elsewhere
        return value

def allocate_time_ms(time_left_ms: int) -> int:
    """Rough budget for this single move. Deliberately conservative: flagging loses
    instantly, using less time than you could does not."""
    budget = time_left_ms / MOVES_DIVISOR + INCREMENT_MS * 0.9 - MOVE_OVERHEAD_MS
    return max(int(budget), MIN_BUDGET_MS)

def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.
 
    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
 
    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.
 
    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
 
    # A move is always ready before we search a single node, so a time-up (or any
    # unexpected exception) at any point still returns something legal.
    best_move_overall = random.choice(legal_moves)
    if len(legal_moves) == 1:
        return best_move_overall.uci()
 
    budget_ms = allocate_time_ms(time_left_ms)
    deadline = time.perf_counter() + budget_ms / 1000
 
    depth = 1
    while depth <= MAX_DEPTH:
        searcher = Searcher(deadline)
        ordered_moves = order_moves(board, legal_moves)
        # Root alpha starts at -inf and beta stays +inf: unlike interior nodes, we
        # never want a cutoff here -- we need every root move's score to pick (and
        # break ties among) the best one. Narrowing alpha as we go still prunes
        # deeper inside each subsequent move's subtree, which is most of the benefit.
        alpha = -math.inf
        best_score = -math.inf
        best_this_depth: list[chess.Move] = []
        try:
            for move in ordered_moves:
                board.push(move)
                try:
                    score = -searcher.negamax(board, depth - 1, -math.inf, -alpha)
                finally:
                    board.pop()
                if score > best_score:
                    best_score = score
                    best_this_depth = [move]
                elif score == best_score:
                    best_this_depth.append(move)
                if score > alpha:
                    alpha = score
        except TimeUp:
            # This depth was cut short: later moves in ordered_moves got less search
            # than earlier ones, so its results aren't comparable. Discard them and
            # keep whatever the last fully-completed depth found.
            break
 
        # This depth finished cleanly -- it's now our best-known move.
        best_move_overall = random.choice(best_this_depth)
 
        if time.perf_counter() >= deadline:
            break
        depth += 1
 
    return best_move_overall.uci()



evaluate(*encode(chess.Board()), 20)
