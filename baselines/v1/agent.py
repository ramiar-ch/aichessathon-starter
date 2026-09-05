"""The submission entrypoint. The platform imports this file and calls get_move."""

import math
import random
import time

import chess
import numpy as np
from numba import njit

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int32)
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
MOBILITY_WEIGHT = 4
MATE = 1e6
 
# --- Piece-square tables -----------------------------------------------------
# Indexed 0 (a1) .. 63 (h8), from White's perspective: a bonus/penalty in centipawns
# for a piece of that type standing on that square, on top of its raw material value.
# For a Black piece, look up square ^ 56 instead (flips the rank, keeps the file --
# the standard trick for reusing a White-oriented table for the mirrored side).
# These are the widely-used "simplified evaluation function" tables
# (chessprogramming.org) -- a reasonable starting point, not hand-tuned for this eval.
PAWN_PST = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.int32)
 
KNIGHT_PST = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
], dtype=np.int32)
 
BISHOP_PST = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
], dtype=np.int32)
 
ROOK_PST = np.array([
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.int32)
 
QUEEN_PST = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
], dtype=np.int32)
 
KING_MID_PST = np.array([
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
], dtype=np.int32)
 
KING_END_PST = np.array([
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50,
], dtype=np.int32)

# Below this much combined non-pawn material (both sides, kings excluded), the king's
# table switches from "stay tucked behind pawns" (middlegame) to "get active, head for
# the centre" (endgame). A coarse on/off switch rather than a smooth taper -- fine for
# now, worth revisiting once the rest of the eval is solid.
ENDGAME_MATERIAL_THRESHOLD = 1700
    
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
def evaluate(pieces: np.ndarray, mine: np.ndarray, white_to_move: bool, mobility: int) -> int:
    material = 0
    pst_score = 0
    non_pawn_material = 0
    white_king_sq = -1
    black_king_sq = -1
 
    for square in range(64):
        piece = pieces[square]
        if piece == 0:
            continue
 
        # mine[square] tells us whose piece it is, relative to the side to move --
        # that's what the +/- sign below needs. is_white tells us its literal colour,
        # which is what PST mirroring needs, and is recoverable from the other two:
        # mine == (colour == white_to_move), so colour == (mine == white_to_move).
        is_white = mine[square] == white_to_move
        sign = 1 if mine[square] else -1
        material += sign * PIECE_VALUE[piece]
 
        if piece == KING:
            # King's PST bonus depends on game phase, which we only know once the
            # whole board has been scanned -- handled after this loop.
            if is_white:
                white_king_sq = square
            else:
                black_king_sq = square
            continue
 
        pst_square = square if is_white else (square ^ 56)
        if piece == PAWN:
            bonus = PAWN_PST[pst_square]
        elif piece == KNIGHT:
            bonus = KNIGHT_PST[pst_square]
        elif piece == BISHOP:
            bonus = BISHOP_PST[pst_square]
        elif piece == ROOK:
            bonus = ROOK_PST[pst_square]
        else:  # QUEEN
            bonus = QUEEN_PST[pst_square]
        pst_score += sign * bonus
 
        if piece != PAWN:
            non_pawn_material += PIECE_VALUE[piece]
 
    endgame = non_pawn_material <= ENDGAME_MATERIAL_THRESHOLD
    king_pst = KING_END_PST if endgame else KING_MID_PST
 
    if white_king_sq >= 0:
        sign = 1 if white_to_move else -1
        pst_score += sign * king_pst[white_king_sq]
    if black_king_sq >= 0:
        sign = -1 if white_to_move else 1
        pst_score += sign * king_pst[black_king_sq ^ 56]
 
    return material + pst_score + MOBILITY_WEIGHT * mobility


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
            return float(evaluate(pieces, mine, board.turn, len(moves)))
 
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



evaluate(*encode(chess.Board()), chess.WHITE, 20)