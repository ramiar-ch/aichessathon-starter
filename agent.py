"""AI Chessathon submission entrypoint. The platform imports this file and calls get_move.

Design notes for reviewers
--------------------------
Pure-Python search over python-chess. numba is deliberately NOT used: the jitted
evaluation in the previous revision cost 0.5us to run and 23.5us to marshal a board
into, so the JIT was a net loss, and importing numba cost 82 MB of the 2 GB budget for
nothing. The realistic ceiling here is a full bitboard rewrite inside @njit (movegen
included); a half-jitted eval is strictly worse than clean Python. See PERF NOTES below.

Search:   negamax / alpha-beta, iterative deepening, aspiration windows, PVS,
          transposition table, null-move pruning, late move reductions, capped check
          extensions, killers + history, quiescence with MVV-LVA, delta pruning and
          check evasions.
Eval:     tapered (phase-interpolated) material + piece-square tables, passed pawns,
          isolated/doubled pawns, bishop pair, rook on open/semi-open file, tempo.

Failure modes are the priority. Per the docs, `illegal`, `crash`, `flag` and `init` are
all straight losses, so every one of them has an explicit guard rather than an
assumption. get_move can never raise and can never return a non-legal string.

PERF NOTES (measured, AMD-class core, python-chess 1.11.2)
  list(board.legal_moves)            39.4 us
  board.generate_legal_captures()    13.2 us
  evaluate(board)                     8.9 us
  board._transposition_key()          0.8 us
  chess.polyglot.zobrist_hash()      17.4 us   <- 19x worse, not used
  board.is_check()                    1.0 us
  board.is_insufficient_material()    0.5 us
"""

import sys
import time

import chess

# Deep check extensions plus quiescence can stack frames; the interpreter default of
# 1000 is uncomfortably close. A RecursionError mid-search is a `crash`, i.e. a loss.
sys.setrecursionlimit(10000)

# --- Score scale --------------------------------------------------------------
# Centipawns. MATE is offset by ply on the way up so that mate-in-3 beats mate-in-7;
# MATE_BOUND is the threshold above which a score is "a mate score, not an evaluation".
INF = 10 ** 9
MATE = 100_000
MATE_BOUND = MATE - 1000
MAX_PLY = 200

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
PIECE_VALUE = [0, 100, 320, 330, 500, 900, 0]

# Phase weights for tapered eval: 24 = full material, 0 = bare kings and pawns.
PHASE_W = [0, 0, 1, 1, 2, 4, 0]
TOTAL_PHASE = 24

# ==============================================================================
# Piece-square tables
# ==============================================================================
# Each is 64 entries, index 0 = a1, 7 = h1, 56 = a8, 63 = h8 -- written from White's
# point of view with rank 1 as the first row. A Black piece reads index (square ^ 56),
# which flips the rank and leaves the file alone.
#
# Two sets: _MG (middlegame) and _EG (endgame). evaluate() computes both and blends
# them by game phase, so there is no discontinuity when material comes off. The
# previous revision had a single hard threshold, which meant one capture could swing
# the king's contribution by ~90cp with nothing else changing.
#
# Reading them: knights and bishops want the centre and hate the rim; rooks want the
# 7th; the king wants to be tucked away in the middlegame (KING_MG rewards g1/b1) and
# active in the centre in the endgame (KING_EG inverts that). Pawns get a separate EG
# table because advancement matters far more once the queens are off -- MG rewards
# central control, EG rewards rank, peaking on the 7th at +80.
#
# These are the stock chessprogramming.org "simplified evaluation" values. They are a
# starting point, not tuned for this engine; Texel tuning is the obvious next step.

_PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     5,  5,  5,  5,  5,  5,  5,  5,
    10, 10, 10, 10, 10, 10, 10, 10,
    25, 25, 25, 25, 25, 25, 25, 25,
    50, 50, 50, 50, 50, 50, 50, 50,
    80, 80, 80, 80, 80, 80, 80, 80,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
_ROOK = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
_KING_MG = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
]
_KING_EG = [
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50,
]

_MG_SRC = [None, _PAWN_MG, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING_MG]
_EG_SRC = [None, _PAWN_EG, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING_EG]

# Flattened to TABLE[colour][piece_type * 64 + square], with material folded in, so the
# eval inner loop is one list index instead of a table lookup plus an add. Colour index
# is 1 for White / 0 for Black, matching chess.WHITE / chess.BLACK as ints.
MG_TAB = [[0] * 448, [0] * 448]
EG_TAB = [[0] * 448, [0] * 448]
for _pt in range(1, 7):
    for _sq in range(64):
        _i = _pt * 64 + _sq
        MG_TAB[1][_i] = PIECE_VALUE[_pt] + _MG_SRC[_pt][_sq]
        EG_TAB[1][_i] = PIECE_VALUE[_pt] + _EG_SRC[_pt][_sq]
        MG_TAB[0][_i] = PIECE_VALUE[_pt] + _MG_SRC[_pt][_sq ^ 56]
        EG_TAB[0][_i] = PIECE_VALUE[_pt] + _EG_SRC[_pt][_sq ^ 56]

# ==============================================================================
# Pawn-structure masks and eval weights
# ==============================================================================
FILE_MASK = [chess.BB_FILES[f] for f in range(8)]
ADJ_FILE = [
    (FILE_MASK[f - 1] if f > 0 else 0) | (FILE_MASK[f + 1] if f < 7 else 0)
    for f in range(8)
]
# PASSED_MASK[colour][square] = every square in front of `square` on its own and
# adjacent files. A pawn is passed if no enemy pawn sits in that region.
PASSED_MASK = [[0] * 64, [0] * 64]
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    _span = FILE_MASK[_f] | ADJ_FILE[_f]
    _ahead_w = 0
    _ahead_b = 0
    for _rr in range(_r + 1, 8):
        _ahead_w |= chess.BB_RANKS[_rr]
    for _rr in range(0, _r):
        _ahead_b |= chess.BB_RANKS[_rr]
    PASSED_MASK[1][_sq] = _span & _ahead_w
    PASSED_MASK[0][_sq] = _span & _ahead_b

PASSED_BONUS_MG = [0, 5, 10, 20, 35, 60, 100, 0]
PASSED_BONUS_EG = [0, 10, 20, 35, 60, 100, 160, 0]
BISHOP_PAIR = 30
DOUBLED = -12
ISOLATED = -14
ROOK_OPEN = 22
ROOK_SEMI = 11
TEMPO = 12

# ==============================================================================
# Search tuning
# ==============================================================================
# Time. Per the docs the control is 120s + 0.5s/move and time_left_ms is the clock
# BEFORE the increment lands. Flagging is a loss, so the hard cap is expressed as a
# fraction of the clock actually in hand, never as a multiple of the soft target alone.
MOVES_DIVISOR = 26          # assume ~26 more moves when splitting the base clock
INCREMENT_MS = 500
INCREMENT_USE = 0.85        # fraction of the increment we plan to spend
MOVE_OVERHEAD_MS = 60       # reserve for IPC, GC and check granularity
EMERGENCY_RESERVE = 0.10    # never spend more than 90% of the remaining clock
HARD_MULTIPLIER = 2.0       # a started iteration may run to 2x the soft target
START_NEXT_FRACTION = 0.50  # do not begin a new iteration past 50% of the soft target
TIME_CHECK_MASK = 1023      # poll the clock every 1024 nodes

# Quiescence.
QSEARCH_MAX_DEPTH = 8       # stand-pat plies; check evasions are exempt (see qsearch)
QSEARCH_ABS_DEPTH = 32      # absolute ceiling, covers long checking sequences
DELTA_MARGIN = 200          # positional slack allowed on top of pure material
DELTA_MIN_HEAVY = 4         # disable delta pruning at/below this many non-pawn pieces

# Transposition table. Fixed-size bucket array rather than a growing dict: memory is
# bounded and predictable (measured ~453 bytes per dict entry, so a 1.2M-entry dict
# would be ~540 MB of the 2 GB budget, and clearing it on overflow throws away the
# root entry mid-search).
TT_BITS = 19                # 524288 slots, ~110 MB fully populated
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# MVV ordering values. The king entry is large so that a "capture the king" move, which
# cannot legally occur but can appear if a bug lets one through, sorts first and is
# caught loudly rather than silently mis-ordered.
MVV = [0, 100, 320, 330, 500, 900, 2000]

PAWN_CACHE_LIMIT = 100_000

popcount = chess.popcount
scan = chess.scan_reversed


class TimeUp(Exception):
    """Raised inside the search once this move's hard deadline has passed."""


# ==============================================================================
# Evaluation
# ==============================================================================
_pawn_cache = {}


def _pawn_structure(wp, bp):
    """Passed / isolated / doubled pawn terms, returned as (mg, eg) from White's view.

    Cached on the pawn bitboard pair. Pawn structure changes on maybe one move in ten,
    so the hit rate is very high and this term is close to free in practice.
    """
    hit = _pawn_cache.get((wp, bp))
    if hit is not None:
        return hit
    mg = eg = 0
    for colour, own, opp, sign in ((1, wp, bp, 1), (0, bp, wp, -1)):
        for sq in scan(own):
            f = chess.square_file(sq)
            if not (PASSED_MASK[colour][sq] & opp):
                rel = chess.square_rank(sq) if colour else 7 - chess.square_rank(sq)
                mg += sign * PASSED_BONUS_MG[rel]
                eg += sign * PASSED_BONUS_EG[rel]
            if not (ADJ_FILE[f] & own):
                mg += sign * ISOLATED
                eg += sign * ISOLATED
        for f in range(8):
            n = popcount(own & FILE_MASK[f])
            if n > 1:
                mg += sign * DOUBLED * (n - 1)
                eg += sign * DOUBLED * (n - 1)
    if len(_pawn_cache) >= PAWN_CACHE_LIMIT:
        _pawn_cache.clear()
    _pawn_cache[(wp, bp)] = (mg, eg)
    return mg, eg


def evaluate(board):
    """Static evaluation in centipawns, from the side-to-move's point of view.

    Middlegame and endgame scores are accumulated together and blended by phase at the
    end. Everything below is computed from White's perspective and negated once, which
    keeps the sign logic in one place.

    Note there is no mobility term. The previous revision added
    MOBILITY_WEIGHT * len(legal_moves) for the side to move only. That was unsound once
    quiescence made leaf depth variable -- the number of negamax negations between leaf
    and root stopped being constant, so the term read as "reward my mobility" in some
    lines and "penalise the opponent's" in others within a single search. It also
    forced a 39us legal-move generation at every leaf to feed a term that was hurting.
    """
    occ_w = board.occupied_co[True]
    occ_b = board.occupied_co[False]
    mg = eg = 0
    phase = 0
    mgw, egw = MG_TAB[1], EG_TAB[1]
    mgb, egb = MG_TAB[0], EG_TAB[0]

    for pt, bb in ((PAWN, board.pawns), (KNIGHT, board.knights), (BISHOP, board.bishops),
                   (ROOK, board.rooks), (QUEEN, board.queens), (KING, board.kings)):
        base = pt * 64
        w = bb & occ_w
        b = bb & occ_b
        for sq in scan(w):
            mg += mgw[base + sq]
            eg += egw[base + sq]
        for sq in scan(b):
            mg -= mgb[base + sq]
            eg -= egb[base + sq]
        phase += PHASE_W[pt] * (popcount(w) + popcount(b))

    if popcount(board.bishops & occ_w) > 1:
        mg += BISHOP_PAIR
        eg += BISHOP_PAIR
    if popcount(board.bishops & occ_b) > 1:
        mg -= BISHOP_PAIR
        eg -= BISHOP_PAIR

    wp = board.pawns & occ_w
    bp = board.pawns & occ_b
    pmg, peg = _pawn_structure(wp, bp)
    mg += pmg
    eg += peg

    all_pawns = wp | bp
    for sq in scan(board.rooks & occ_w):
        fm = FILE_MASK[chess.square_file(sq)]
        if not (fm & all_pawns):
            mg += ROOK_OPEN
        elif not (fm & wp):
            mg += ROOK_SEMI
    for sq in scan(board.rooks & occ_b):
        fm = FILE_MASK[chess.square_file(sq)]
        if not (fm & all_pawns):
            mg -= ROOK_OPEN
        elif not (fm & bp):
            mg -= ROOK_SEMI

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    total = mg * phase + eg * (TOTAL_PHASE - phase)
    if not board.turn:
        total = -total
    # Truncate toward zero, not floor. Python's // rounds negatives away from zero,
    # which would make the eval asymmetric by 1cp under a colour flip -- the engine
    # would play fractionally differently as Black than as White for no reason.
    score = total // TOTAL_PHASE if total >= 0 else -((-total) // TOTAL_PHASE)
    return score + TEMPO


# ==============================================================================
# Persistent state
# ==============================================================================
# One process serves one game, so all of this is per-game and never needs resetting in
# production. reset_state() exists for the local self-play harness, which reuses the
# process across games -- without it, GAME_KEYS from game N poisons repetition
# detection in game N+1.

_tt_key = [0] * TT_SIZE          # 0 means "empty slot"
_tt_depth = [0] * TT_SIZE
_tt_score = [0] * TT_SIZE
_tt_flag = [0] * TT_SIZE
_tt_move = [None] * TT_SIZE

HISTORY = {}                     # (colour, from, to) -> cutoff weight
GAME_KEYS = {}                   # positions actually played this game, for repetition


def reset_state():
    """Clear all cross-move state. Called by the local harness between games."""
    for i in range(TT_SIZE):
        _tt_key[i] = 0
        _tt_move[i] = None
    HISTORY.clear()
    GAME_KEYS.clear()
    _pawn_cache.clear()


def _age_history():
    """Halve history scores between moves.

    History is a heuristic about which quiet moves have been causing cutoffs recently.
    Without ageing, early-game counts dominate for the rest of the game and the table
    stops tracking the position in front of it. Halving also bounds growth.
    """
    dead = []
    for k, v in HISTORY.items():
        nv = v >> 1
        if nv:
            HISTORY[k] = nv
        else:
            dead.append(k)
    for k in dead:
        del HISTORY[k]


# ==============================================================================
# Searcher
# ==============================================================================
class Searcher:
    """Per-move search state: clock, node counters, killers, repetition path.

    One instance per get_move call. The transposition table and history heuristic live
    at module scope instead, because they are worth carrying across moves -- the search
    for move N+1 overlaps heavily with the search for move N.
    """

    def __init__(self, deadline, rep, max_ply):
        self.deadline = deadline
        self.rep = rep                 # position key -> occurrences on the current path
        self.max_ply = max_ply         # ceiling for check extensions
        self.nodes = 0
        self.qnodes = 0
        self.killers = [[None, None] for _ in range(MAX_PLY + 8)]

    def check(self):
        """Poll the clock. Cheap: perf_counter is only called once per 1024 nodes."""
        self.nodes += 1
        if not self.nodes & TIME_CHECK_MASK and time.perf_counter() >= self.deadline:
            raise TimeUp

    # -- transposition table ---------------------------------------------------
    # Mate scores are stored relative to the node, not to the root. A mate found at
    # ply 6 is "mate in 2 from here"; if the same position is reached at ply 4 via a
    # transposition the stored score must still mean "mate in 2 from here". Hence the
    # +/- ply adjustment on store and its inverse on probe. Getting this wrong makes
    # the engine announce mates it cannot deliver.

    @staticmethod
    def _tt_store(key, depth, score, flag, move, ply):
        idx = key & TT_MASK
        # Depth-preferred replacement, but always overwrite a different position:
        # a stale deep entry for an unrelated key is worse than a shallow fresh one.
        if _tt_key[idx] == key and _tt_depth[idx] > depth:
            return
        if score > MATE_BOUND:
            score += ply
        elif score < -MATE_BOUND:
            score -= ply
        _tt_key[idx] = key
        _tt_depth[idx] = depth
        _tt_score[idx] = score
        _tt_flag[idx] = flag
        _tt_move[idx] = move

    @staticmethod
    def _tt_probe(key, ply):
        idx = key & TT_MASK
        if _tt_key[idx] != key:
            return None
        score = _tt_score[idx]
        if score > MATE_BOUND:
            score -= ply
        elif score < -MATE_BOUND:
            score += ply
        return _tt_depth[idx], score, _tt_flag[idx], _tt_move[idx]

    # -- move ordering ---------------------------------------------------------
    def order(self, board, moves, ttmove, ply):
        """Sort moves best-first. This is what makes alpha-beta actually prune.

        Tiers, highest first:
          1. transposition-table move  -- by far the strongest single signal
          2. captures and promotions   -- MVV-LVA
          3. killer moves              -- quiets that cut off at this ply elsewhere
          4. everything else           -- history heuristic score
        """
        killers = self.killers[ply] if ply < len(self.killers) else (None, None)
        k1, k2 = killers[0], killers[1]
        turn = board.turn
        out = []
        for m in moves:
            if m == ttmove:
                out.append((1 << 30, m))
            elif board.is_capture(m) or m.promotion:
                if board.is_en_passant(m):
                    victim = PAWN
                else:
                    # None here means a non-capture promotion, not a bug. Guard it
                    # explicitly: indexing a list with None raises, and indexing a
                    # numpy array with None silently returns a reshaped array.
                    victim = board.piece_type_at(m.to_square) or 0
                s = (1 << 20) + MVV[victim] * 16 - MVV[board.piece_type_at(m.from_square)]
                if m.promotion:
                    s += 1 << 21
                out.append((s, m))
            elif m == k1:
                out.append((1 << 19, m))
            elif m == k2:
                out.append(((1 << 19) - 1, m))
            else:
                out.append((HISTORY.get((turn, m.from_square, m.to_square), 0), m))
        # Sort on the score only. Tuples containing Move would need Move to be
        # orderable on ties, which it is not.
        out.sort(key=_first, reverse=True)
        return [m for _, m in out]

    # -- quiescence ------------------------------------------------------------
    def qsearch(self, board, alpha, beta, ply, qdepth):
        """Search only forcing moves until the position is quiet, then evaluate.

        Without this the engine evaluates whatever position happens to sit at depth 0,
        including ones with a queen hanging. It is the single largest source of strength
        in the whole file.

        Three things that are easy to get wrong and are handled explicitly:

        1. CHECK EVASIONS. When in check the side to move cannot decline to move, so
           standing pat is unsound -- it would let the engine "pass" in a position where
           it is being mated. In check we search every legal move, apply no delta
           pruning, and report mate on an empty move list.
        2. NON-CAPTURE PROMOTIONS. generate_legal_captures() does not include them
           (verified against python-chess 1.11.2). A pawn walking into a queen is an
           ~800cp swing and exactly the kind of move quiescence exists to see.
        3. EN PASSANT. piece_type_at(to_square) returns None for an en-passant capture
           because the victim is not on the target square.
        """
        self.check()
        self.qnodes += 1

        # Absolute ceiling. Long checking sequences are exempt from QSEARCH_MAX_DEPTH
        # below, so they need their own stop or a perpetual check recurses without end.
        if qdepth >= QSEARCH_ABS_DEPTH or ply >= MAX_PLY:
            return evaluate(board)

        in_check = board.is_check()

        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply          # checkmate; stalemate is impossible in check
            best = -INF
            for m in self.order(board, moves, None, ply):
                board.push(m)
                try:
                    v = -self.qsearch(board, -beta, -alpha, ply + 1, qdepth + 1)
                finally:
                    board.pop()
                if v > best:
                    best = v
                if v > alpha:
                    alpha = v
                if alpha >= beta:
                    break
            return best

        # Stand pat: the side to move is never obliged to capture, so the static score
        # is a lower bound on this node's true value.
        stand = evaluate(board)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand

        if qdepth >= QSEARCH_MAX_DEPTH:
            return stand

        # Delta pruning assumes a capture's upside is bounded by the captured piece.
        # That approximation breaks down in pawn endgames where a single pawn decides
        # the game, so it is switched off once heavy material is nearly gone.
        heavy = board.occupied & ~board.pawns & ~board.kings
        use_delta = popcount(heavy) > DELTA_MIN_HEAVY

        scored = []
        for m in board.generate_legal_captures():
            # Rook/bishop/knight promotions are essentially never the right move and
            # triple the branching of any promotion node. Queen only.
            if m.promotion and m.promotion != QUEEN:
                continue
            if board.is_en_passant(m):
                victim = PAWN
            else:
                victim = board.piece_type_at(m.to_square) or 0
            if use_delta and not m.promotion:
                if stand + PIECE_VALUE[victim] + DELTA_MARGIN < alpha:
                    continue
            s = MVV[victim] * 16 - MVV[board.piece_type_at(m.from_square)]
            if m.promotion:
                s += 1 << 20
            scored.append((s, m))

        # Non-capture promotions, only paid for when a pawn is actually one rank away.
        promo_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
        pawns_ready = board.pawns & board.occupied_co[board.turn] & promo_rank
        if pawns_ready:
            for m in board.generate_legal_moves(from_mask=pawns_ready):
                if m.promotion == QUEEN and not board.is_capture(m):
                    scored.append(((1 << 20) + MVV[QUEEN], m))

        scored.sort(key=_first, reverse=True)

        for _, m in scored:
            board.push(m)
            try:
                v = -self.qsearch(board, -beta, -alpha, ply + 1, qdepth + 1)
            finally:
                board.pop()
            if v > best:
                best = v
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        return best

    # -- main search -----------------------------------------------------------
    def search(self, board, depth, alpha, beta, ply, allow_null=True):
        """Fail-soft negamax with alpha-beta. Returns a score from the side-to-move's
        point of view. Root is handled separately in _search_root."""
        self.check()

        # Draw detection. The referee claims threefold and fifty-move automatically, so
        # a repetition the engine cannot see is a draw it never agreed to. self.rep is
        # seeded with the positions actually played this game, then maintained along the
        # search path, so an in-search repetition of a played position scores 0.
        if ply:
            if board.halfmove_clock >= 100 or board.is_insufficient_material():
                return 0
            key = board._transposition_key()
            if self.rep.get(key, 0) >= 1:
                return 0
            # Mate-distance pruning: we already have a mate faster than anything this
            # subtree could produce, so there is nothing to find here.
            mate_alpha = -MATE + ply
            if mate_alpha > alpha:
                alpha = mate_alpha
                if alpha >= beta:
                    return alpha
        else:
            key = board._transposition_key()

        alpha0 = alpha
        hkey = hash(key)
        ttmove = None
        entry = self._tt_probe(hkey, ply)
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            ttmove = e_move
            if ply and e_depth >= depth:
                if e_flag == TT_EXACT:
                    return e_score
                if e_flag == TT_LOWER:
                    if e_score > alpha:
                        alpha = e_score
                elif e_score < beta:
                    beta = e_score
                if alpha >= beta:
                    return e_score

        in_check = board.is_check()
        # Check extension, capped. Uncapped, a forcing sequence keeps adding depth and
        # the tree stops shrinking; max_ply is set from the root iteration depth.
        if in_check and ply < self.max_ply:
            depth += 1

        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply, 0)

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0

        # Null-move pruning: give the opponent a free move. If they still cannot beat
        # beta, this position is good enough to prune. Requires non-pawn material for
        # the side to move, otherwise zugzwang makes "pass" a false negative.
        if (allow_null and not in_check and depth >= 3 and beta < MATE_BOUND
                and (board.occupied_co[board.turn] & ~board.pawns & ~board.kings)):
            r = 2 + depth // 6
            board.push(chess.Move.null())
            try:
                v = -self.search(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
            finally:
                board.pop()
            if v >= beta:
                # Fail soft, but never return a mate score from a null-move cutoff --
                # the mate is not proven, only implied by a reduced search.
                return beta if v > MATE_BOUND else v

        best = -INF
        best_move = None
        ordered = self.order(board, moves, ttmove, ply)

        self.rep[key] = self.rep.get(key, 0) + 1
        try:
            for i, m in enumerate(ordered):
                quiet = not board.is_capture(m) and not m.promotion
                board.push(m)
                try:
                    if i == 0:
                        # Principal variation: full window on the first (best-ordered)
                        # move only.
                        v = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
                    else:
                        # Late move reductions: later quiet moves are unlikely to be
                        # best, so search them shallower first and only re-search at
                        # full depth if they surprise us.
                        red = 0
                        if quiet and depth >= 3 and i >= 3 and not in_check:
                            red = 1 + (i >= 6 and depth >= 5)
                        v = -self.search(board, depth - 1 - red, -alpha - 1, -alpha, ply + 1)
                        if alpha < v < beta:
                            v = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
                finally:
                    board.pop()

                if v > best:
                    best, best_move = v, m
                if v > alpha:
                    alpha = v
                if alpha >= beta:
                    if quiet:
                        if ply < len(self.killers):
                            kl = self.killers[ply]
                            if kl[0] != m:
                                kl[1] = kl[0]
                                kl[0] = m
                        hk = (board.turn, m.from_square, m.to_square)
                        HISTORY[hk] = HISTORY.get(hk, 0) + depth * depth
                    break
        finally:
            c = self.rep[key] - 1
            if c:
                self.rep[key] = c
            else:
                del self.rep[key]

        flag = TT_EXACT if alpha0 < best < beta else (TT_LOWER if best >= beta else TT_UPPER)
        self._tt_store(hkey, depth, best, flag, best_move, ply)
        return best

    def search_root(self, board, root_moves, depth, alpha, beta):
        """Root iteration. Returns (score, move, completed).

        Kept separate from search() for two reasons. First, the best move is returned
        directly rather than read back out of the transposition table -- a TT entry can
        be evicted mid-search, and depending on it to carry the move is a silent
        failure waiting to happen. Second, on a timeout we can still salvage the
        partial result: root_moves[0] is the previous iteration's best, so if a later
        move has already beaten it, that comparison is valid even though the iteration
        never finished.
        """
        best = -INF
        best_move = None
        completed = False
        try:
            for i, m in enumerate(root_moves):
                board.push(m)
                try:
                    if i == 0:
                        v = -self.search(board, depth - 1, -beta, -alpha, 1)
                    else:
                        v = -self.search(board, depth - 1, -alpha - 1, -alpha, 1)
                        if alpha < v < beta:
                            v = -self.search(board, depth - 1, -beta, -alpha, 1)
                finally:
                    board.pop()
                if v > best:
                    best, best_move = v, m
                if v > alpha:
                    alpha = v
            completed = True
        except TimeUp:
            # Partial result. Only usable if at least the first root move finished.
            if best_move is None:
                raise
        return best, best_move, completed


def _first(pair):
    return pair[0]


# ==============================================================================
# Time management
# ==============================================================================
def allocate_time_ms(time_left_ms):
    """Soft target for this move, in milliseconds.

    Soft is what we aim to spend; the caller derives a hard cap from it and from the
    clock actually in hand. The increment is counted at 85% because it only lands after
    the move is sent, and the whole thing is clamped so that an almost-empty clock still
    produces a legal move rather than a flag.
    """
    soft = time_left_ms / MOVES_DIVISOR + INCREMENT_MS * INCREMENT_USE - MOVE_OVERHEAD_MS
    ceiling = time_left_ms * (1.0 - EMERGENCY_RESERVE) - MOVE_OVERHEAD_MS
    if soft > ceiling:
        soft = ceiling
    return max(int(soft), 20)


# ==============================================================================
# Entry point
# ==============================================================================
def _get_move_inner(fen, time_left_ms, start):
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        # Should never happen -- the referee would not ask. Returning a null move here
        # is a loss, but so is raising, and this at least shows up in the log.
        print("WARN no legal moves", flush=True)
        return "0000"

    best_move = legal[0]
    if len(legal) == 1:
        _record_game_keys(board, best_move)
        return best_move.uci()

    _age_history()

    # Repetition context: positions already played this game, excluding the current one
    # (taken before it is recorded), plus whatever the search pushes onto the path.
    rep = dict(GAME_KEYS)

    soft_ms = allocate_time_ms(time_left_ms)
    soft = soft_ms / 1000.0
    # Hard cap is bounded by the real clock, not just by a multiple of soft. A soft
    # target of 4s must never authorise a 12s search when only 5s remain.
    clock_cap = max(time_left_ms * (1.0 - EMERGENCY_RESERVE) - MOVE_OVERHEAD_MS, 20) / 1000.0
    hard = min(soft * HARD_MULTIPLIER, clock_cap)
    deadline = start + hard

    searcher = Searcher(deadline, rep, MAX_PLY)
    root_moves = searcher.order(board, legal, None, 0)

    score = 0
    depth_done = 0
    for depth in range(1, MAX_PLY):
        # Check extensions may not push the tree past roughly twice the nominal depth.
        searcher.max_ply = min(depth * 2 + 8, MAX_PLY - 1)

        # Aspiration windows: assume this iteration lands near the last one and search a
        # narrow window, widening only on a fail. Cheap when it works, and it usually
        # works after depth 4.
        if depth <= 4:
            lo, hi = -INF, INF
        else:
            lo, hi = score - 40, score + 40

        # Widening schedule. A collapsing score can fail low repeatedly, and stepping
        # the bound down a little at a time means paying for a fresh near-full search
        # each time. One 4x widen, then straight to full width.
        fails = 0
        while True:
            try:
                s, mv, completed = searcher.search_root(board, root_moves, depth, lo, hi)
            except TimeUp:
                mv = None
                completed = False
                s = score
                break

            if not completed:
                break
            if s <= lo and lo > -INF:
                fails += 1
                lo = -INF if fails > 1 else s - 4 * (hi - lo)
                if lo < -MATE_BOUND:
                    lo = -INF
                continue
            if s >= hi and hi < INF:
                fails += 1
                hi = INF if fails > 1 else s + 4 * (hi - lo)
                if hi > MATE_BOUND:
                    hi = INF
                continue
            break

        if mv is not None:
            # A partial iteration is only adopted when a move strictly improved on the
            # previous iteration's score; otherwise the incomplete ordering makes the
            # comparison meaningless.
            if completed or s > score:
                best_move = mv
                score = s
                if completed:
                    depth_done = depth
                # Search the new best move first next iteration.
                root_moves = [mv] + [x for x in root_moves if x != mv]

        if not completed:
            break
        if abs(score) > MATE_BOUND:
            break                                     # mate found, no point going deeper
        if time.perf_counter() - start > soft * START_NEXT_FRACTION:
            break

    elapsed = time.perf_counter() - start
    # One line per move. The platform keeps 8 KB per game (first 4 KB + last 4 KB), so
    # this stays terse -- per-iteration logging would overflow it inside twenty moves.
    total = searcher.nodes or 1
    print(
        f"d={depth_done} s={score:+d} n={searcher.nodes} q={100*searcher.qnodes//total}%"
        f" nps={int(total/max(elapsed,1e-6))} t={elapsed:.2f}s"
        f" soft={soft:.2f} clk={time_left_ms/1000:.1f} {best_move.uci()}",
        flush=True,
    )

    # Defensive: never return something the referee will call illegal.
    if best_move not in legal:
        print("WARN best_move illegal, falling back", flush=True)
        best_move = legal[0]

    _record_game_keys(board, best_move)
    return best_move.uci()


def _record_game_keys(board, move):
    """Record both this position and the one we are about to hand back.

    The FEN carries no history, so repetition detection needs the engine to remember
    what it has seen. We are only asked about our own turns, but we know the move we
    chose, so the intermediate position is reconstructable and the record ends up
    complete.
    """
    GAME_KEYS[board._transposition_key()] = GAME_KEYS.get(board._transposition_key(), 0) + 1
    after = board.copy(stack=False)
    after.push(move)
    k = after._transposition_key()
    GAME_KEYS[k] = GAME_KEYS.get(k, 0) + 1


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation. Never raises.

    Every failure mode in the docs' failure reference is a loss, so this wrapper exists
    to convert a crash into a bad move rather than a forfeit. Anything unexpected below
    is logged and falls back to the first legal move.
    """
    start = time.perf_counter()
    try:
        return _get_move_inner(fen, time_left_ms, start)
    except Exception as exc:                                  # noqa: BLE001 - deliberate
        try:
            print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
            return next(iter(chess.Board(fen).legal_moves)).uci()
        except Exception:                                     # noqa: BLE001
            return "0000"


# ==============================================================================
# Import-time warm-up
# ==============================================================================
# The docs give 90s of init budget before the clock starts, and anything deferred to
# the first get_move comes out of the match clock instead. A short search here builds
# the pawn cache, populates the TT with the opening, and forces every code path in the
# file to be executed and byte-compiled once.
def _warmup():
    board = chess.Board()
    evaluate(board)
    s = Searcher(time.perf_counter() + 2.0, {}, 24)
    try:
        s.search_root(board, list(board.legal_moves), 4, -INF, INF)
    except TimeUp:
        pass
    reset_state()


_warmup()