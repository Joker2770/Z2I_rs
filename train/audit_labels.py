"""Audit the self-play label stream before it can waste a training round.

The learner's worst failures are silent. A round whose games abort (no winner, so every ply
is labelled `v = 0`), or whose winner is always the same colour, still trains, still lowers
the value loss, and only shows up many rounds later as a value head that answers "whose turn
is it" instead of "who is winning" -- which the weight probe then reports as a value that
leans on the redundant colour plane (`COLOUR_PLANE_LEAN_VALUE_SHIFT` in `src/probe.rs`) and
which its value criteria reject. This command prints the label
statistics that make those datasets visible before 20 rounds are spent on them.

It also verifies the labels against the **final position** of every game: the stored `v` and
`colour` only say who the engine thought won, while the board says whether a winning line is
really there, whether it belongs to that colour, and whether the rule that was in force would
have accepted it (per `src/free_style.rs`, `standard.rs`, `renju.rs`, `caro.rs`). The label
statistics alone cannot catch that class of error, because the labels are self-consistent by
construction -- they all come from the same wrong number.

File layout (see `src/play.rs`, mirrors `learner.py::load_samples`):

    step      i32
    rule      i32              (absent in legacy files, which default to FreeStyle = 0)
    boards    step * n * n     i32
    pi        step * n * n     f32
    v         step             i32      (+1/-1 for the side to move, 0 if the game had no winner)
    colour    step             i32      (+1 Black / -1 White side to move)
    last_move step             i32

Usage:
    cd train && python3 audit_labels.py                 # newest 40 files of ../build
    python3 audit_labels.py --dir ../build --all         # every file in the window folders
    python3 audit_labels.py --dir ../build --expect-rule 1
    python3 audit_labels.py --self-test                  # no disk access beyond a temp dir
"""

import argparse
import math
import os
import shutil
import struct
import sys
import tempfile
from collections import Counter

# folder order used by `learner.py`: the live window first, then the replay backup
WINDOW_FOLDERS = ("data", "data_backup")
ARCHIVE_FOLDER = "data_archive"
# the learner's replay window: examples_buffer_max_len * games_per_iter (common.py)
REPLAY_WINDOW_FILES = 320
# a ply whose label is this close to the colour alone cannot be learned from ch0/ch1
COLOUR_ONLY_MEAN = 0.9
# below this many plies per side, per-colour means say nothing (one short game saturates them)
COLOUR_ONLY_MIN_PLIES = 20
# below this many games no winner rate means anything: a single game always has one winner
MIN_GAMES_FOR_VERDICT = 5
# rule flag bits (src/rule.rs): 1 = Standard (exactly five), 4 = Renju, 0 = FreeStyle,
# 8 = Caro; a combination requires every sub-rule to agree (src/gomoku.rs)
RULE_STANDARD = 0b0001
RULE_RENJU = 0b0100
RULE_CARO = 0b1000
BLACK, WHITE = 1, -1
# the four line directions every judge in src/ scans: horizontal, vertical, both diagonals
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
# board size of the synthetic fixtures: wide enough to fit a five plus a blocker at each end,
# which the Caro rule-semantics tests below need; shared by write_game and self_test so the two
# cannot drift apart
SELFTEST_BOARD = 8


def parse_batch_id(file_name):
    """Parse batch_id from a data file name `data_{batch_id}_{hex}`, else None."""
    parts = os.path.basename(file_name).split("_")
    if len(parts) >= 3 and parts[0] == "data":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def select_files(build_dir, folders, newest):
    """Return the newest `newest` data files across `folders` (None = all).

    Ordering matches the learner's replay window (mtime, then batch id, descending), so the
    audit describes the same files the next training round would consume.
    """
    candidates = []
    for folder in folders:
        directory = os.path.join(build_dir, folder)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            batch_id = parse_batch_id(name)
            candidates.append((mtime, batch_id if batch_id is not None else -1, path))
    candidates.sort(reverse=True)
    if newest is not None:
        candidates = candidates[:newest]
    return [path for _, _, path in candidates]


def read_game(path, board, verify_termination=True):
    """Parse one data file.

    Returns `(summary, reason)`: `summary` is a dict for a readable file, `reason` a string
    when the file is skipped. Only the pi/v/colour blocks are read; the board block is
    skipped, because the audit is about labels, not positions.
    """
    size = os.path.getsize(path)
    plane = board * board
    bytes_per_step = plane * 4 + plane * 4 + 3 * 4
    with open(path, "rb") as binfile:
        header = binfile.read(4)
        if len(header) < 4:
            return None, "empty file"
        step = int.from_bytes(header, byteorder="little", signed=True)
        payload = step * bytes_per_step
        old_expected = 4 + payload
        new_expected = old_expected + 4
        if step <= 0 or size < old_expected:
            return None, f"incomplete (step={step}, size={size}, expected={new_expected})"
        if size >= new_expected:
            rule = int.from_bytes(binfile.read(4), byteorder="little", signed=True)
        else:
            rule = 0
        # the board blocks are only needed for the winner check, and only the final snapshot
        # is used: the policy block starts right after them
        board_start = binfile.tell()
        binfile.seek(board_start + step * plane * 4, os.SEEK_SET)
        pi = struct.unpack(f"<{step * plane}f", binfile.read(step * plane * 4))
        v = struct.unpack(f"<{step}i", binfile.read(step * 4))
        color = struct.unpack(f"<{step}i", binfile.read(step * 4))
        last_moves = struct.unpack(f"<{step}i", binfile.read(step * 4))
        last_position = None
        if verify_termination:
            # the snapshot of the last stored ply is the position *before* the move that
            # decided the game (`play.rs` stores the pre-move board at every ply), which is
            # what the termination check reads; the deciding move itself is not stored
            binfile.seek(board_start + (step - 1) * plane * 4, os.SEEK_SET)
            last_position = struct.unpack(f"<{plane}i", binfile.read(plane * 4))
    summary = {
        "path": path,
        "step": step,
        "rule": rule,
        "v": v,
        "color": color,
        "last_moves": last_moves,
        "termination_check": termination_check(last_position, v, color, board, rule, step)
        if verify_termination else (None, "unchecked", "not checked"),
        # only Caro can legitimately play past a five, so only Caro files are asked about it
        "caro_blocked_fives": caro_blocked_fives(last_position, board)
        if verify_termination and rule & RULE_CARO and last_position is not None
        else [],
        "pi_entropy": [],
        "pi_top1": [],
        "pi_support": [],
        "pi_sum": [],
    }
    for ply in range(step):
        row = pi[ply * plane : (ply + 1) * plane]
        summary["pi_sum"].append(math.fsum(row))
        summary["pi_top1"].append(max(row))
        summary["pi_support"].append(sum(1 for value in row if value > 1e-6))
        summary["pi_entropy"].append(
            -math.fsum(value * math.log(value) for value in row if value > 0.0)
        )
    return summary, None


def classify_winner(v, color):
    """Derive the winner of one game from its labels, plus the ply-level disagreements.

    `v` is the result for the side to move at that ply (`play.rs`: `v = colour * win`), so a
    ply with `v == 0` in a game that has a winner means the label stream is inconsistent.
    """
    decided = [ply for ply in range(len(v)) if v[ply] != 0]
    if not decided:
        return None, 0
    votes = [color[ply] if v[ply] > 0 else -color[ply] for ply in decided]
    winner = votes[0]
    disagreements = sum(1 for vote in votes if vote != winner)
    return winner, disagreements


def colour_name(colour):
    """Board value to a readable name (`play.rs` writes +1 for Black, -1 for White)."""
    if colour == BLACK:
        return "Black"
    if colour == WHITE:
        return "White"
    return "empty"


def run_bounds(board, board_size, index, direction):
    """Describe the same-colour run through `index` along `direction`.

    Returns `(length, before, after)`, where `before`/`after` are the board values just outside
    the run -- `None` where the board edge stops it, because `caro.rs` treats the edge as a wall
    rather than as a block (its `WIN_SHAPES` holds `{3,1,1,1,1,1,2}`, a five the edge abuts).
    A stone with no neighbour of its own colour in this direction yields a length of 1.
    """
    stone = board[index]
    if stone == 0:
        return 0, None, None
    d_row, d_col = direction
    row, col = divmod(index, board_size)
    length = 1
    bounds = {}
    for sign in (-1, 1):
        r = row + sign * d_row
        c = col + sign * d_col
        while 0 <= r < board_size and 0 <= c < board_size and board[r * board_size + c] == stone:
            length += 1
            r += sign * d_row
            c += sign * d_col
        bounds[sign] = (
            board[r * board_size + c] if 0 <= r < board_size and 0 <= c < board_size else None
        )
    return length, bounds[-1], bounds[1]


def describe_run(board, board_size, index, direction, length):
    """Readable location of a run, e.g. `row 3 col 4 (horizontal, 5 in a row)` (0-based)."""
    d_row, d_col = direction
    name = {
        (0, 1): "horizontal",
        (1, 0): "vertical",
        (1, 1): "diagonal",
        (1, -1): "antidiagonal",
    }[direction]
    row, col = divmod(index, board_size)
    stone = board[index]
    # walk back to the run's first stone so the printed cell is where the line starts
    while (
        0 <= row - d_row < board_size
        and 0 <= col - d_col < board_size
        and board[(row - d_row) * board_size + col - d_col] == stone
    ):
        row -= d_row
        col -= d_col
    return f"row {row} col {col} ({name}, {length} in a row)"


def caro_accepts_run(length, colour, before, after):
    """Whether `caro.rs` counts a run of `length` as a win for `colour`.

    Caro is five-or-more like FreeStyle with one extra refusal: a five blocked by an opponent
    stone at *both* ends is not a win, which is why `WIN_SHAPES` deliberately omits
    `{2,1,1,1,1,1,2}` (`oxxxxxo`). The board edge is a wall, not a block -- `{3,1,1,1,1,1,2}`,
    a five with the edge on one side, is in the table.
    """
    if length < 5:
        return False
    return not (length == 5 and before == -colour and after == -colour)


def is_winning_run(length, colour, rule, before, after):
    """Whether a run of `length` bounded by `before`/`after` is a win for `colour`.

    Mirrors the judges in src/: five or more wins in FreeStyle (`free_style.rs` counts `>= 4`
    neighbours) and in Caro, minus the doubly blocked five (`caro_accepts_run`); Standard counts
    exactly five (`standard.rs` uses `== 4`, so an overline is not a win); Renju counts exactly
    five for Black, whose overline is a forbidden shape instead, while White's overline still
    wins (`renju.rs`). A combination (e.g. Standard|Caro = 9) needs every bit to accept, so both
    the exact-five test and the Caro test are applied.
    """
    if (rule & RULE_STANDARD) or (rule & RULE_RENJU and colour == BLACK):
        if length != 5:
            return False
    elif length < 5:
        return False
    if rule & RULE_CARO and not caro_accepts_run(length, colour, before, after):
        return False
    return True


def ends_game(length, colour, rule, before, after):
    """Whether a run means the game was already over: a win, or a loss on `colour`'s own move.

    Renju is the only rule that ends a game on a shape that is not a win for the mover: Black's
    overline (six or more) is a forbidden move, so it loses the game for Black (`renju.rs`).
    """
    if is_winning_run(length, colour, rule, before, after):
        return True
    return bool(rule & RULE_RENJU) and colour == BLACK and length >= 6


def line_windows(board_size, length=5):
    """Every contiguous `length`-cell line on the board, as index tuples (4 directions)."""
    for d_row, d_col in DIRECTIONS:
        for row in range(board_size):
            for col in range(board_size):
                end_row = row + (length - 1) * d_row
                end_col = col + (length - 1) * d_col
                if not (0 <= end_row < board_size and 0 <= end_col < board_size):
                    continue
                yield tuple(
                    (row + step * d_row) * board_size + (col + step * d_col)
                    for step in range(length)
                )


def deciding_run(board, board_size, colour, rule):
    """A run of `colour` that already ended the game (readable label), else None.

    Per direction rather than per longest direction: a five that Caro refuses may still be a
    stone of another, winning line, and the two need not lie in the same direction.
    """
    for index, stone in enumerate(board):
        if stone != colour:
            continue
        for direction in DIRECTIONS:
            length, before, after = run_bounds(board, board_size, index, direction)
            if ends_game(length, colour, rule, before, after):
                return describe_run(board, board_size, index, direction, length)
    return None


def wins_at(board, board_size, index, colour, rule):
    """Whether the stone `colour` has just placed at `index` completes a winning line."""
    for direction in DIRECTIONS:
        length, before, after = run_bounds(board, board_size, index, direction)
        if is_winning_run(length, colour, rule, before, after):
            return True
    return False


def has_winning_reply(board, board_size, colour, rule):
    """Whether `colour` to move wins by filling a gap: a 5-window holding four of its stones
    and one empty cell, where landing the stone really does win under `rule`.

    Counting the four is not enough under Caro: `oxxxx_o` holds a four, but its only completion
    is `oxxxxxo`, a five the rule refuses, so it is no threat -- and a file that labels a win
    from such a position is exactly the poison this command hunts.
    """
    board = list(board)
    for window in line_windows(board_size):
        stones = [board[index] for index in window]
        if stones.count(colour) != 4 or stones.count(0) != 1:
            continue
        gap = window[stones.index(0)]
        board[gap] = colour
        won = wins_at(board, board_size, gap, colour, rule)
        board[gap] = 0
        if won:
            return True
    return False


def caro_blocked_fives(board, board_size):
    """Every five on the board that Caro refuses because the opponent blocks both ends.

    These explain a Caro game that played past a five without anyone having misplayed, so the
    report lists them instead of leaving the reader to guess why the game did not end.
    """
    blocked = set()
    for index, stone in enumerate(board):
        if stone == 0:
            continue
        for direction in DIRECTIONS:
            length, before, after = run_bounds(board, board_size, index, direction)
            if length == 5 and before == -stone and after == -stone:
                blocked.add(describe_run(board, board_size, index, direction, length))
    return sorted(blocked)


def termination_check(board, v, color, board_size, rule, step):
    """Check a game's ending against the position it was decided from.

    `play.rs` stores the position **before** the move about to be played at every ply, so the
    move that decided a game is not in the file -- but the ending is still pinned down:

    * the side to move at the last stored ply plays the deciding move, so a positive label
      there has to be a win from that position: some gap-fill has to complete a line the rule
      accepts. Anything else means the labels and the positions disagree, which poisons every
      target in the file while leaving the aggregate statistics perfectly self-consistent;
    * a negative label there means the mover lost **on their own move**, which only Renju
      allows (Black's forbidden move ends the game with White winning). Under any other rule
      that cannot happen, so it is reported;
    * and neither side may already hold a deciding line before that move, which would mean the
      game ran past its end; a Caro five blocked by an opponent stone at both ends is
      deliberately not such a line, so it is listed as `caro_blocked_fives` instead of being
      reported as a failure (see `caro_accepts_run`).

    Returns (True/False/None, kind, detail). `kind` is "five", "forbidden" (Renju only) or
    "unchecked"; None means there was nothing to check.
    """
    if step < 2 or color[-1] not in (BLACK, WHITE):
        return None, "unchecked", "no side to move to check"
    label = v[-1]
    if label == 0:
        return None, "unchecked", "the last ply carries no label"
    mover = color[-1]
    for colour in (BLACK, WHITE):
        found = deciding_run(board, board_size, colour, rule)
        if found:
            return (False, "unchecked",
                    f"{colour_name(colour)} already holds a deciding line before the last "
                    f"move ({found}), so the game should have ended earlier")
    if label > 0:
        if not has_winning_reply(board, board_size, mover, rule):
            return (False, "unchecked",
                    f"labelled {colour_name(mover)} win on the last move, but the position "
                    f"before it lets {colour_name(mover)} complete no winning line")
        return True, "five", f"{colour_name(mover)} completes a five"
    if rule & RULE_RENJU and mover == BLACK:
        return (True, "forbidden",
                "forbidden move (Renju Black): the mover loses on their own move")
    return (False, "unchecked",
            f"labelled {colour_name(mover)} loss on their own last move, but rule {rule} has "
            f"no way for a player to lose on their own move")


def audit(games):
    """Aggregate per-file summaries into the report the command prints."""
    report = {
        "games": len(games),
        "rules": {},
        "lengths": [],
        "winners": {1: 0, -1: 0, None: 0},
        "v_counts": {1: 0, -1: 0, 0: 0},
        "v_by_color": {1: [], -1: []},
        "plies": 0,
        "disagreements": 0,
        "termination_games": 0,
        "termination_failures": [],
        "forbidden_endings": 0,
        "caro_blocked": [],
        "pi_entropy": [],
        "pi_top1": [],
        "pi_support": [],
        "pi_unnormalized": 0,
    }
    for game in games:
        report["rules"][game["rule"]] = report["rules"].get(game["rule"], 0) + 1
        report["lengths"].append(game["step"])
        winner, disagreements = classify_winner(game["v"], game["color"])
        report["winners"][winner] += 1
        report["disagreements"] += disagreements
        verified, kind, detail = game["termination_check"]
        if verified is not None:
            report["termination_games"] += 1
            if verified:
                if kind == "forbidden":
                    report["forbidden_endings"] += 1
            else:
                report["termination_failures"].append((game["path"], detail))
        if game["caro_blocked_fives"]:
            report["caro_blocked"].append((game["path"], game["caro_blocked_fives"][0]))
        for ply in range(game["step"]):
            report["plies"] += 1
            value = game["v"][ply]
            colour = game["color"][ply]
            report["v_counts"][value if value in (1, -1) else 0] += 1
            if colour in report["v_by_color"]:
                report["v_by_color"][colour].append(value)
        report["pi_entropy"].extend(game["pi_entropy"])
        report["pi_top1"].extend(game["pi_top1"])
        report["pi_support"].extend(game["pi_support"])
        report["pi_unnormalized"] += sum(
            1 for total in game["pi_sum"] if abs(total - 1.0) > 1e-3
        )
    return report


def mean(values):
    return math.fsum(values) / len(values) if values else float("nan")


def print_report(report, skipped, candidates, expect_rule, board):
    """Print the human-readable audit and return whether the labels are degenerate.

    Degenerate means the label stream cannot teach a value function: no winner in most games
    (every ply labelled 0), a single-colour winner rate, a `v` that is saturated with the side
    to move -- the last one is precisely the ch3 shortcut a value head collapses into, which
    `COLOUR_PLANE_LEAN_VALUE_SHIFT` in `src/probe.rs` reports on the weight side (as an
    advisory line; the probe's value criteria are what reject such a weight) -- or labels
    that the final position contradicts.
    """
    games = report["games"]
    print(f"# label audit ({board}x{board}, {candidates} candidate file(s))")
    print(f"parsed: {games} game(s), skipped: {len(skipped)}")
    for path, reason in skipped:
        print(f"  skip {os.path.basename(path)}: {reason}")
    if not games:
        print("no readable games: nothing to say about the labels (run --self-test)")
        return False

    lengths = report["lengths"]
    print(
        f"games: {games}   length min {min(lengths)} / mean {mean(lengths):.1f} / "
        f"max {max(lengths)}"
    )
    rules = ", ".join(f"{flag}={count}" for flag, count in sorted(report["rules"].items()))
    print(f"rule flag(s): {rules}" + (f" (expected {expect_rule})" if expect_rule is not None else ""))
    if expect_rule is not None and any(flag != expect_rule for flag in report["rules"]):
        print(
            "  WARNING: mixed rule flags; the learner skips every file whose header does not "
            "match config['rule'] (train/common.py)"
        )

    winners = report["winners"]
    black_share = 100.0 * winners[1] / games
    white_share = 100.0 * winners[-1] / games
    none_share = 100.0 * winners[None] / games
    print(
        f"winner: Black {winners[1]} ({black_share:.1f}%)   White {winners[-1]} "
        f"({white_share:.1f}%)   none/aborted {winners[None]} ({none_share:.1f}%)"
    )

    plies = report["plies"]
    counts = report["v_counts"]
    print(
        f"v per ply: +1 {counts[1]} ({100.0 * counts[1] / plies:.1f}%)   -1 {counts[-1]} "
        f"({100.0 * counts[-1] / plies:.1f}%)   other {counts[0]} "
        f"({100.0 * counts[0] / plies:.1f}%)"
    )
    by_color = report["v_by_color"]
    black_mean = mean(by_color[1])
    white_mean = mean(by_color[-1])
    print(
        f"v by side to move (the ch3-shortcut test): Black {black_mean:+.3f} "
        f"({len(by_color[1])} plies)   White {white_mean:+.3f} ({len(by_color[-1])} plies)"
    )

    checked = report["termination_games"]
    if checked:
        failed = len(report["termination_failures"])
        print(
            f"termination check: {checked - failed}/{checked} game(s) end the way their labels "
            f"say (only the pre-move position is stored, so a five is checked as the four "
            f"that completes it)"
        )
        if report["forbidden_endings"]:
            print(
                f"  note: {report['forbidden_endings']} game(s) ended with the mover losing on "
                f"their own move -- Renju's forbidden-move loss, i.e. Black played an illegal "
                f"shape (those plies teach \"Black to move means Black is losing\")"
            )
        for file_path, detail in report["termination_failures"][:5]:
            print(f"  {os.path.basename(file_path)}: {detail}")
        if report["caro_blocked"]:
            print(
                f"  note: {len(report['caro_blocked'])} game(s) hold a five Caro does not count "
                f"(the run is blocked by an opponent stone at both ends, `oxxxxxo`), which is "
                f"why those games played on instead of ending:"
            )
            for file_path, detail in report["caro_blocked"][:5]:
                print(f"    {os.path.basename(file_path)}: {detail}")

    degenerate = False
    if report["termination_failures"]:
        print(
            "  DEGENERATE: the stored labels disagree with the positions they were decided "
            "from, so those files poison every target in them -- inspect them before training "
            "on this window"
        )
        degenerate = True
    if report["disagreements"]:
        print(
            f"  WARNING: {report['disagreements']} ply label(s) contradict their own game's "
            f"winner: the v stream is inconsistent (play.rs writes v = colour * win)"
        )
        degenerate = True
    # the two verdicts below are rates, so they need a sample before they mean anything
    if games < MIN_GAMES_FOR_VERDICT:
        print(
            f"note: only {games} game(s) readable; the winner-rate verdicts need at least "
            f"{MIN_GAMES_FOR_VERDICT} (a single game is always won by one colour)"
        )
    if games >= MIN_GAMES_FOR_VERDICT and none_share > 50.0:
        print(
            "  DEGENERATE: most games have no winner, so most plies are labelled 0 and the "
            "value head can only learn \"nobody is winning\" -- check the self-play "
            "termination path before training on this window"
        )
        degenerate = True
    single_colour = max(black_share, white_share)
    if games >= MIN_GAMES_FOR_VERDICT and single_colour >= 99.0:
        print(
            f"  DEGENERATE: {single_colour:.1f}% of games are won by one colour. The label is "
            f"then a function of the side to move, which a value head can fit from the "
            f"constant colour plane alone -- that is the collapse the weight probe reports"
        )
        degenerate = True
    if (
        games >= MIN_GAMES_FOR_VERDICT
        and abs(black_mean) >= COLOUR_ONLY_MEAN
        and abs(white_mean) >= COLOUR_ONLY_MEAN
        and black_mean * white_mean < 0.0
        and len(by_color[1]) >= COLOUR_ONLY_MIN_PLIES
        and len(by_color[-1]) >= COLOUR_ONLY_MIN_PLIES
    ):
        print(
            "  the two means are saturated, so `v` is predictable from the constant colour "
            "plane alone (ch3); that is the shortcut a value head collapses into -- check "
            "the winner line above"
        )

    print(
        f"pi: mean entropy {mean(report['pi_entropy']):.2f} nats, mean top-1 "
        f"{mean(report['pi_top1']):.3f}, mean support {mean(report['pi_support']):.1f} "
        f"move(s), non-normalized ply rows {report['pi_unnormalized']}"
    )
    if report["pi_top1"] and mean(report["pi_top1"]) < 0.05:
        print(
            "  WARNING: the search targets are nearly uniform, so the policy has almost "
            "nothing to learn from (raise the simulation count or start from a stronger net)"
        )
    return degenerate


def read_window(paths, board, verify_termination=True):
    """Read every file in `paths`, returning `(games, skipped)`."""
    games, skipped = [], []
    for path in paths:
        game, reason = read_game(path, board, verify_termination)
        if game is None:
            skipped.append((path, reason))
        else:
            games.append(game)
    return games, skipped


# ------------------------------------------------------------------ self-test


def write_game(path, rule, plies, board=None, last_move=None):
    """Write one synthetic data file.

    `plies` is a list of (colour, v, pi row). `board`/`last_move` describe the **final**
    position and are replicated into every ply's snapshot: only the last snapshot is read
    (by the winner check), and the feature content is irrelevant to a label audit.
    """
    size = SELFTEST_BOARD
    plane = size * size
    stones = list(board) if board is not None else [0] * plane
    assert len(stones) == plane, stones
    last_moves = [last_move if last_move is not None else -1] * len(plies)
    with open(path, "wb") as binfile:
        binfile.write(struct.pack("<i", len(plies)))
        binfile.write(struct.pack("<i", rule))
        for _ in plies:
            binfile.write(struct.pack(f"<{plane}i", *stones))
        for _, _, row in plies:
            assert len(row) == plane
            binfile.write(struct.pack(f"<{plane}f", *row))
        binfile.write(struct.pack(f"<{len(plies)}i", *[v for _, v, _ in plies]))
        binfile.write(struct.pack(f"<{len(plies)}i", *[c for c, _, _ in plies]))
        binfile.write(struct.pack(f"<{len(plies)}i", *last_moves))


def self_test():
    """Round-trip synthetic windows: a learnable one, then the degenerate variants."""
    temp = tempfile.mkdtemp(prefix="audit_labels_")
    try:
        board = SELFTEST_BOARD
        plane = board * board
        one_hot = [0.8, 0.1, 0.1] + [0.0] * (plane - 3)
        expected_entropy = -math.fsum(
            value * math.log(value) for value in one_hot if value > 0.0
        )
        # final positions the termination check reads: `play.rs` stores the position *before*
        # the move that decided the game, so a five shows up as the four that completes it
        black_four = [BLACK] * 4 + [0] * (plane - 4)
        white_four = [WHITE] * 4 + [0] * (plane - 4)
        black_five = [BLACK] * 5 + [0] * (plane - 5)

        normal_dir = os.path.join(temp, "normal")
        os.makedirs(normal_dir)
        # Black wins: the labels follow the mover, and the last stored ply has the winner to
        # move -- that is who plays the deciding move, which the file does not store
        black_win = [(1, 1, one_hot), (-1, -1, one_hot), (1, 1, one_hot),
                     (-1, -1, one_hot), (1, 1, one_hot)]
        # White wins: the same shape with Black moving first
        white_win = [(1, -1, one_hot), (-1, 1, one_hot), (1, -1, one_hot), (-1, 1, one_hot)]
        # the mover loses on their own last move: only Renju allows that (forbidden move)
        mover_loses = [(-1, 1, one_hot), (1, -1, one_hot)]
        aborted = [(1, 0, one_hot), (-1, 0, one_hot)]
        # a learnable window: both colours win, and one game is undecided
        game_id = 0
        for _ in range(3):
            write_game(os.path.join(normal_dir, f"data_{game_id}_deadbeef"), 1, black_win,
                       board=black_four)
            game_id += 16
        for _ in range(2):
            write_game(os.path.join(normal_dir, f"data_{game_id}_cafebabe"), 1, white_win,
                       board=white_four)
            game_id += 16
        write_game(os.path.join(normal_dir, f"data_{game_id}_0badc0de"), 1, aborted)
        # a truncated file: it claims four plies and holds none
        with open(os.path.join(normal_dir, "data_9999_partial"), "wb") as binfile:
            binfile.write(struct.pack("<i", 4))

        selected = select_files(normal_dir, ("",), None)
        assert len(selected) == 7, selected
        games, skipped = read_window(selected, board)
        assert len(games) == 6, games
        assert len(skipped) == 1, skipped
        assert skipped[0][1].startswith("incomplete"), skipped

        report = audit(games)
        assert report["games"] == 6, report
        assert report["rules"] == {1: 6}, report["rules"]
        # three Black wins, two White wins, one game with no winner at all
        assert report["winners"] == {1: 3, -1: 2, None: 1}, report["winners"]
        # 25 plies: +1 thirteen times, -1 ten times, 0 twice
        assert report["v_counts"] == {1: 13, -1: 10, 0: 2}, report["v_counts"]
        assert report["plies"] == 25, report["plies"]
        assert report["disagreements"] == 0, report
        # both colours move, and every label matches its own game's winner (counted rather
        # than listed: the file order follows mtime, which is not part of the contract)
        assert Counter(report["v_by_color"][1]) == Counter({1: 9, -1: 4, 0: 1}), report["v_by_color"]
        assert Counter(report["v_by_color"][-1]) == Counter({1: 4, -1: 6, 0: 1}), report["v_by_color"]
        # every labelled win really is a four waiting to complete a five, and the undecided
        # game has nothing to check
        assert report["termination_games"] == 5, report["termination_games"]
        assert report["termination_failures"] == [], report["termination_failures"]
        assert report["forbidden_endings"] == 0, report["forbidden_endings"]
        assert report["pi_unnormalized"] == 0, report
        # the rows are written as f32, so the entropy matches the Python floats to f32
        # precision rather than exactly
        assert abs(mean(report["pi_entropy"]) - expected_entropy) < 1e-6, report["pi_entropy"]
        assert print_report(report, skipped, len(selected), 1, board) is False

        # one colour winning everything is the degenerate case this command exists for
        black_dir = os.path.join(temp, "black_only")
        os.makedirs(black_dir)
        for game_id in range(0, 6 * 16, 16):
            write_game(os.path.join(black_dir, f"data_{game_id}_beefbeef"), 1, black_win,
                       board=black_four)
        black_games, black_skipped = read_window(
            select_files(black_dir, ("",), None), board
        )
        black_report = audit(black_games)
        assert black_report["winners"] == {1: 6, -1: 0, None: 0}, black_report["winners"]
        assert black_report["termination_failures"] == [], black_report["termination_failures"]
        assert print_report(black_report, black_skipped, 6, 1, board) is True

        # ... and so is a window where nothing is ever decided
        abort_dir = os.path.join(temp, "aborted")
        os.makedirs(abort_dir)
        for game_id in range(0, 6 * 16, 16):
            write_game(os.path.join(abort_dir, f"data_{game_id}_f00df00d"), 1, aborted)
        abort_games, abort_skipped = read_window(
            select_files(abort_dir, ("",), None), board
        )
        abort_report = audit(abort_games)
        assert abort_report["winners"][None] == 6, abort_report["winners"]
        assert abort_report["termination_games"] == 0, abort_report["termination_games"]
        assert print_report(abort_report, abort_skipped, 6, 1, board) is True

        # the termination check must catch labels the position contradicts
        bad_dir = os.path.join(temp, "bad_labels")
        os.makedirs(bad_dir)
        # labelled Black win, but the four waiting on the board is White's
        write_game(os.path.join(bad_dir, "data_0_aaaaaaaa"), 1, black_win, board=white_four)
        # a five is already on the board, so the game ran past its end
        write_game(os.path.join(bad_dir, "data_16_bbbbbbbb"), 1, black_win, board=black_five)
        # a mover cannot lose on their own move under FreeStyle
        write_game(os.path.join(bad_dir, "data_32_eeeeeeee"), 0, mover_loses, board=black_four)
        bad_games, bad_skipped = read_window(select_files(bad_dir, ("",), None), board)
        bad_report = audit(bad_games)
        assert bad_report["termination_games"] == 3, bad_report["termination_games"]
        assert len(bad_report["termination_failures"]) == 3, bad_report["termination_failures"]
        assert print_report(bad_report, bad_skipped, 3, None, board) is True

        # ... while the same ending is Renju's forbidden-move loss, which is legitimate
        ok_dir = os.path.join(temp, "rule_semantics")
        os.makedirs(ok_dir)
        write_game(os.path.join(ok_dir, "data_0_cccccccc"), 4, mover_loses, board=black_four)
        write_game(os.path.join(ok_dir, "data_16_dddddddd"), 1, black_win, board=black_four)
        ok_games, ok_skipped = read_window(select_files(ok_dir, ("",), None), board)
        ok_report = audit(ok_games)
        assert ok_report["termination_games"] == 2, ok_report["termination_games"]
        assert ok_report["termination_failures"] == [], ok_report["termination_failures"]
        assert ok_report["forbidden_endings"] == 1, ok_report["forbidden_endings"]
        assert print_report(ok_report, ok_skipped, 2, None, board) is False

        # the rule model the checks rest on, pinned directly: every five wins except the doubly
        # blocked one, and an overline always does (the enumeration the Rust test
        # `five_in_a_row_wins_unless_both_ends_are_blocked` runs against the real judge)
        for colour in (BLACK, WHITE):
            for before in (0, colour, -colour, None):
                for after in (0, colour, -colour, None):
                    assert is_winning_run(5, colour, RULE_CARO, before, after) == (
                        not (before == -colour and after == -colour)
                    ), (colour, before, after)
                    assert is_winning_run(6, colour, RULE_CARO, before, after), (colour, before)

        # Caro: `oxxxxxo` -- a five blocked by an opponent stone at both ends -- is deliberately
        # not a win, so a Caro game may legitimately play past one. Reading it as a deciding line
        # is what turns a healthy Caro window into a false DEGENERATE verdict.
        caro_blocked = [0] * plane
        for col, value in enumerate([BLACK, WHITE, WHITE, WHITE, WHITE, WHITE, BLACK, 0]):
            caro_blocked[1 * board + col] = value
        # the mover's own four (row 3, cols 0-3), the position it is labelled to win from
        for col in range(4):
            caro_blocked[3 * board + col] = BLACK
        # the same four, but its only completion is a five Caro refuses (`oxxxx_o`)
        caro_dead_four = list(caro_blocked)
        for col, value in enumerate([BLACK, WHITE, WHITE, WHITE, WHITE, 0, BLACK, 0]):
            caro_dead_four[1 * board + col] = value

        caro_dir = os.path.join(temp, "caro_legal")
        os.makedirs(caro_dir)
        write_game(os.path.join(caro_dir, "data_0_car0car0"), 8, black_win, board=caro_blocked)
        # the same four under FreeStyle is a real threat (a five need not be unblocked there),
        # so this file's labels hold up too
        write_game(os.path.join(caro_dir, "data_16_0dd0dd0d"), 0, white_win, board=caro_dead_four)
        caro_games, caro_skipped = read_window(select_files(caro_dir, ("",), None), board)
        caro_report = audit(caro_games)
        assert caro_report["termination_failures"] == [], caro_report["termination_failures"]
        # the ignored five is listed once, with the line's location, not once per stone in it
        assert len(caro_report["caro_blocked"]) == 1, caro_report["caro_blocked"]
        assert caro_report["caro_blocked"][0][1].startswith("row 1 col 1"), caro_report["caro_blocked"]
        assert print_report(caro_report, caro_skipped, 2, None, board) is False

        # ... but the exemption is exactly one shape, so everything else is still caught
        caught_dir = os.path.join(temp, "caro_still_caught")
        os.makedirs(caught_dir)
        # an unblocked Caro five ends the game, so this one ran past its end
        caro_open = list(caro_blocked)
        caro_open[1 * board + 0] = 0
        write_game(os.path.join(caught_dir, "data_0_0badc0de"), 8, black_win, board=caro_open)
        # FreeStyle counts the blocked five, so the same board is a failure under that rule
        write_game(os.path.join(caught_dir, "data_16_cafebabe"), 0, black_win, board=caro_blocked)
        # under Caro a dead four completes no winning five, so "White wins on the last move"
        # disagrees with the position it was decided from
        write_game(os.path.join(caught_dir, "data_32_deadf00d"), 8, white_win,
                   board=caro_dead_four)
        caught_games, caught_skipped = read_window(select_files(caught_dir, ("",), None), board)
        caught_report = audit(caught_games)
        assert len(caught_report["termination_failures"]) == 3, caught_report["termination_failures"]
        assert caught_report["caro_blocked"] == [], caught_report["caro_blocked"]
        assert print_report(caught_report, caught_skipped, 3, None, board) is True

        # a window too small to judge must say so instead of crying degeneracy
        one_dir = os.path.join(temp, "one_game")
        os.makedirs(one_dir)
        write_game(os.path.join(one_dir, "data_0_11111111"), 1, white_win, board=white_four)
        one_games, one_skipped = read_window(select_files(one_dir, ("",), None), board)
        assert print_report(audit(one_games), one_skipped, 1, 1, board) is False

        print("self-test: OK")
        return 0
    finally:
        shutil.rmtree(temp, ignore_errors=True)


# ------------------------------------------------------------------ cli


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit self-play data labels (winner, game length, v by side to move).",
    )
    parser.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      os.pardir, "build"),
                        help="training work dir containing data/ and data_backup/ "
                             "(default: ../build)")
    parser.add_argument("--files", type=int, default=40,
                        help=f"how many of the newest files to audit (default 40; the learner's "
                             f"replay window is {REPLAY_WINDOW_FILES})")
    parser.add_argument("--all", action="store_true", help="audit every file in the window")
    parser.add_argument("--include-archive", action="store_true",
                        help="also read data_archive/ (the learner's short-window fallback)")
    parser.add_argument("--no-termination-check", action="store_true",
                        help="skip verifying that each game ends the way its labels say")
    parser.add_argument("--expect-rule", type=int, default=None,
                        help="the rule flag config['rule'] expects; mixed headers are reported")
    parser.add_argument("--board", type=int, default=15, help="board size (default 15)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic round-trip check, no disk access beyond a temp dir")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.files <= 0 and not args.all:
        parser.error("--files must be positive (use --all for every file)")

    folders = list(WINDOW_FOLDERS)
    if args.include_archive:
        folders.append(ARCHIVE_FOLDER)
    build_dir = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(build_dir):
        sys.exit(f"work dir not found: {build_dir}")
    newest = None if args.all else args.files
    selected = select_files(build_dir, folders, newest)
    if not selected:
        print(f"no data files under {build_dir}/{{{','.join(folders)}}}: nothing to audit")
        return 0

    games, skipped = read_window(selected, args.board,
                                 verify_termination=not args.no_termination_check)
    degenerate = print_report(
        audit(games), skipped, len(selected), args.expect_rule, args.board
    )
    if degenerate:
        print("verdict: DEGENERATE labels -- fix the generation side before training on this window")
        return 1
    print("verdict: labels look learnable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
