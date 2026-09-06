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

INF = 10 ** 9
MATE = 100_000
MATE_BOUND = MATE - 1000

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
PIECE_VALUE = [0, 100, 320, 330, 500, 900, 0]
PHASE_W = [0, 0, 1, 1, 2, 4, 0]
TOTAL_PHASE = 24

# --- Piece-square tables (PSTs) ----------------------------------------------
# Each table is a 64-element array giving a centipawn bonus/penalty for placing that
# piece type on that square. Added on top of the piece's raw material value.
#
# LAYOUT: index 0 = a1, 7 = h1, 8 = a2, ..., 63 = h8. Written from White's
# perspective, rank 1 (White's back rank) at the top of each literal array,
# rank 8 at the bottom. For a Black piece, look up (square ^ 56), which flips
# the rank so Black's back rank maps to the same table rows as White's.
#
# HOW TO READ THEM:
#   PAWN_PST   - penalises flank pawns on rank 2 (-20 for d/e), rewards centre
#                control on ranks 4-5 (+25), and heavily rewards advanced pawns
#                on ranks 6-7 (+50). Rank 1 and 8 are 0 (pawns can't stand there).
#   KNIGHT_PST - strong centre preference (+20 on d4/d5/e4/e5), heavy edge
#                penalty (-50 on corners). "A knight on the rim is dim."
#   BISHOP_PST - rewards diagonals and central squares, penalises edges.
#   ROOK_PST   - bonus on the 7th rank (+10, where rooks are powerful) and a
#                small preference for the d/e files (+5 on rank 1).
#   QUEEN_PST  - mild centre preference; avoids the corners and edges.
#   KING_MID   - strongly rewards castled positions (g1/b1 = +30), heavily
#                penalises a king in the centre or advanced (-40 to -50).
#   KING_END   - opposite of middlegame: rewards centralisation (+40 on d4/e4),
#                penalises edges/corners. The king should be active in the endgame.
#
# WEAKNESS: only one pawn table (middlegame-oriented). In the endgame, advanced
# pawns close to promotion should be worth much more. A separate PAWN_EG table
# with a tapered phase interpolation would fix this. Also, these are the stock
# chessprogramming.org values -- tuning them (e.g. Texel tuning) would help.
# These are the widely-used "simplified evaluation function" tables
# (chessprogramming.org) -- a reasonable starting point, not hand-tuned for this eval.

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

# Flattened value+PST tables: TABLE[colour][pt * 64 + square].
MG_TAB = [[0] * 448, [0] * 448]
EG_TAB = [[0] * 448, [0] * 448]
for _pt in range(1, 7):
    for _sq in range(64):
        MG_TAB[1][_pt * 64 + _sq] = PIECE_VALUE[_pt] + _MG_SRC[_pt][_sq]
        EG_TAB[1][_pt * 64 + _sq] = PIECE_VALUE[_pt] + _EG_SRC[_pt][_sq]
        MG_TAB[0][_pt * 64 + _sq] = PIECE_VALUE[_pt] + _MG_SRC[_pt][_sq ^ 56]
        EG_TAB[0][_pt * 64 + _sq] = PIECE_VALUE[_pt] + _EG_SRC[_pt][_sq ^ 56]
 
FILE_MASK = [chess.BB_FILES[f] for f in range(8)]
ADJ_FILE = [
    (FILE_MASK[f - 1] if f > 0 else 0) | (FILE_MASK[f + 1] if f < 7 else 0)
    for f in range(8)
]
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
 
popcount = chess.popcount
scan = chess.scan_reversed
 
_pawn_cache = {}
 
 
def _pawn_structure(wp, bp):
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
    if len(_pawn_cache) > 200000:
        _pawn_cache.clear()
    _pawn_cache[(wp, bp)] = (mg, eg)
    return mg, eg
 
 
def evaluate(board):
    occ_w = board.occupied_co[True]
    occ_b = board.occupied_co[False]
    mg = eg = 0
    phase = 0
    mgw = MG_TAB[1]
    egw = EG_TAB[1]
    mgb = MG_TAB[0]
    egb = EG_TAB[0]
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
    score = (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    if not board.turn:
        score = -score
    return score + TEMPO
 
 
class TimeUp(Exception):
    pass
 
 
TT = {}
HISTORY = {}
GAME_KEYS = {}
 
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
MVV = [0, 100, 320, 330, 500, 900, 2000]
 
 
class Searcher:
    def __init__(self, deadline, rep):
        self.deadline = deadline
        self.nodes = 0
        self.rep = rep
        self.killers = [[None, None] for _ in range(128)]
        self.stop = False
 
    def check(self):
        self.nodes += 1
        if not self.nodes & 2047 and time.perf_counter() >= self.deadline:
            raise TimeUp
 
    def qsearch(self, board, alpha, beta):
        self.check()
        stand = evaluate(board)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
        caps = []
        for m in board.generate_legal_captures():
            victim = PAWN if board.is_en_passant(m) else board.piece_type_at(m.to_square)
            s = MVV[victim] * 16 - MVV[board.piece_type_at(m.from_square)]
            if m.promotion:
                s += 10000 + PIECE_VALUE[m.promotion]
            if stand + MVV[victim] + 200 < alpha and not m.promotion:
                continue  # delta pruning
            caps.append((s, m))
        caps.sort(key=lambda t: t[0], reverse=True)
        for _, m in caps:
            board.push(m)
            try:
                v = -self.qsearch(board, -beta, -alpha)
            finally:
                board.pop()
            if v > best:
                best = v
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        return best
 
    def order(self, board, moves, ttmove, ply):
        k1, k2 = self.killers[ply]
        out = []
        for m in moves:
            if m == ttmove:
                out.append((1 << 30, m))
                continue
            if board.is_capture(m) or m.promotion:
                victim = PAWN if board.is_en_passant(m) else (board.piece_type_at(m.to_square) or 0)
                s = 1 << 20
                s += MVV[victim] * 16 - MVV[board.piece_type_at(m.from_square)]
                if m.promotion:
                    s += 1 << 21
                out.append((s, m))
            elif m == k1:
                out.append((1 << 19, m))
            elif m == k2:
                out.append(((1 << 19) - 1, m))
            else:
                out.append((HISTORY.get((board.turn, m.from_square, m.to_square), 0), m))
        out.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in out]
 
    def search(self, board, depth, alpha, beta, ply, allow_null=True):
        self.check()
        if ply:
            if board.halfmove_clock >= 100 or board.is_insufficient_material():
                return 0
            key0 = board._transposition_key()
            if self.rep.get(key0, 0) >= 1:
                return 0
        alpha0 = alpha
        if ply:
            mate_a = -MATE + ply
            if mate_a > alpha:
                alpha = mate_a
                if alpha >= beta:
                    return alpha
 
        key = board._transposition_key()
        ttmove = None
        entry = TT.get(key)
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            ttmove = e_move
            if ply and e_depth >= depth:
                if e_flag == TT_EXACT:
                    return e_score
                if e_flag == TT_LOWER and e_score > alpha:
                    alpha = e_score
                elif e_flag == TT_UPPER and e_score < beta:
                    beta = e_score
                if alpha >= beta:
                    return e_score
 
        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self.qsearch(board, alpha, beta)
 
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
 
        # Null-move pruning
        if (allow_null and not in_check and depth >= 3 and beta < MATE_BOUND
                and (board.occupied_co[board.turn] & ~board.pawns & ~board.kings)):
            r = 2 + depth // 6
            board.push(chess.Move.null())
            try:
                v = -self.search(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
            finally:
                board.pop()
            if v >= beta:
                return beta
 
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
                        v = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
                    else:
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
                        kl = self.killers[ply]
                        if kl[0] != m:
                            kl[1] = kl[0]
                            kl[0] = m
                        k = (board.turn, m.from_square, m.to_square)
                        HISTORY[k] = HISTORY.get(k, 0) + depth * depth
                    break
        finally:
            c = self.rep[key] - 1
            if c:
                self.rep[key] = c
            else:
                del self.rep[key]
 
        flag = TT_EXACT if alpha0 < best < beta else (TT_LOWER if best >= beta else TT_UPPER)
        prev = TT.get(key)
        if prev is None or prev[0] <= depth:
            if len(TT) > 1_200_000:
                TT.clear()
            TT[key] = (depth, best, flag, best_move)
        return best
 
 
_prev_keys = []
 
 
def allocate_time_ms(time_left_ms):
    soft = time_left_ms / 26 + 450 - 40
    return max(int(soft), 20)
 
 
def get_move(fen, time_left_ms):
    start = time.perf_counter()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    best_move = legal[0]
    if len(legal) == 1:
        return best_move.uci()
 
    rep = dict(GAME_KEYS)
    key_now = board._transposition_key()
    GAME_KEYS[key_now] = GAME_KEYS.get(key_now, 0) + 1
 
    soft = allocate_time_ms(time_left_ms) / 1000
    hard = min(soft * 3.0, max(time_left_ms / 1000 - 0.2, 0.02))
    deadline = start + hard
 
    searcher = Searcher(deadline, rep)
    prev_score = 0
    for depth in range(1, 64):
        try:
            if depth <= 4:
                score = searcher.search(board, depth, -INF, INF, 0)
            else:
                window = 40
                while True:
                    a, b = prev_score - window, prev_score + window
                    score = searcher.search(board, depth, a, b, 0)
                    if a < score < b:
                        break
                    window *= 4
                    if window > 2000:
                        score = searcher.search(board, depth, -INF, INF, 0)
                        break
        except TimeUp:
            break
        entry = TT.get(key_now)
        if entry is not None and entry[3] is not None:
            best_move = entry[3]
        prev_score = score
        if abs(score) > MATE_BOUND:
            break
        elapsed = time.perf_counter() - start
        if elapsed > soft * 0.55:
            break
 
    after = board.copy(stack=False)
    after.push(best_move)
    k2 = after._transposition_key()
    GAME_KEYS[k2] = GAME_KEYS.get(k2, 0) + 1
    return best_move.uci()
 