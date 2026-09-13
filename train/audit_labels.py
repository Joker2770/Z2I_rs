"""Audit the self-play label stream before it can waste a training round.

The learner's worst failures are silent. A round whose games abort (no winner, so every ply
is labelled `v = 0`), or whose winner is always the same colour, still trains, still lowers
the value loss, and only shows up many rounds later as a value head that answers "whose turn
is it" instead of "who is winning" -- which the weight probe then reports as a colour-plane
collapse (`COLOUR_COLLAPSE_VALUE_SHIFT` in `src/probe.rs`). This command prints the label
statistics that make those datasets visible before 20 rounds are spent on them.

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


def read_game(path, board):
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
        # boards are not needed for a label audit: jump straight to the policy block, whose
        # offset depends on whether this file carries the rule field
        binfile.seek(binfile.tell() + step * plane * 4, os.SEEK_SET)
        pi = struct.unpack(f"<{step * plane}f", binfile.read(step * plane * 4))
        v = struct.unpack(f"<{step}i", binfile.read(step * 4))
        color = struct.unpack(f"<{step}i", binfile.read(step * 4))
    summary = {
        "path": path,
        "step": step,
        "rule": rule,
        "v": v,
        "color": color,
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
    (every ply labelled 0), a single-colour winner rate, or a `v` that is saturated with the
    side to move -- the last one is precisely the ch3 shortcut a value head collapses into,
    and what `COLOUR_COLLAPSE_VALUE_SHIFT` in `src/probe.rs` reports on the weight side.
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

    degenerate = False
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


def read_window(paths, board):
    """Read every file in `paths`, returning `(games, skipped)`."""
    games, skipped = [], []
    for path in paths:
        game, reason = read_game(path, board)
        if game is None:
            skipped.append((path, reason))
        else:
            games.append(game)
    return games, skipped


# ------------------------------------------------------------------ self-test


def write_game(path, rule, plies):
    """Write one synthetic data file: `plies` is a list of (colour, v, pi row)."""
    board = 3  # the self-test only needs the layout, not a legal game
    plane = board * board
    with open(path, "wb") as binfile:
        binfile.write(struct.pack("<i", len(plies)))
        binfile.write(struct.pack("<i", rule))
        for _ in plies:
            binfile.write(struct.pack(f"<{plane}i", *([0] * plane)))
        for _, _, row in plies:
            assert len(row) == plane
            binfile.write(struct.pack(f"<{plane}f", *row))
        binfile.write(struct.pack(f"<{len(plies)}i", *[v for _, v, _ in plies]))
        binfile.write(struct.pack(f"<{len(plies)}i", *[c for c, _, _ in plies]))
        binfile.write(struct.pack(f"<{len(plies)}i", *[-1] * len(plies)))


def self_test():
    """Round-trip synthetic windows: a learnable one, then the degenerate variants."""
    temp = tempfile.mkdtemp(prefix="audit_labels_")
    try:
        board = 3
        plane = board * board
        one_hot = [0.8, 0.1, 0.1] + [0.0] * (plane - 3)
        expected_entropy = -math.fsum(
            value * math.log(value) for value in one_hot if value > 0.0
        )

        normal_dir = os.path.join(temp, "normal")
        os.makedirs(normal_dir)
        black_win = [(1, 1, one_hot), (-1, -1, one_hot), (1, 1, one_hot), (-1, -1, one_hot)]
        white_win = [(1, -1, one_hot), (-1, 1, one_hot), (1, -1, one_hot)]
        aborted = [(1, 0, one_hot), (-1, 0, one_hot)]
        # a learnable window: both colours win, and one game is undecided
        game_id = 0
        for _ in range(3):
            write_game(os.path.join(normal_dir, f"data_{game_id}_deadbeef"), 1, black_win)
            game_id += 16
        for _ in range(2):
            write_game(os.path.join(normal_dir, f"data_{game_id}_cafebabe"), 1, white_win)
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
        # 20 plies: +1 eight times, -1 ten times, 0 twice
        assert report["v_counts"] == {1: 8, -1: 10, 0: 2}, report["v_counts"]
        assert report["plies"] == 20, report["plies"]
        assert report["disagreements"] == 0, report
        # both colours move, and every label matches its own game's winner (counted rather
        # than listed: the file order follows mtime, which is not part of the contract)
        assert Counter(report["v_by_color"][1]) == Counter({1: 6, -1: 4, 0: 1}), report["v_by_color"]
        assert Counter(report["v_by_color"][-1]) == Counter({1: 2, -1: 6, 0: 1}), report["v_by_color"]
        assert report["pi_unnormalized"] == 0, report
        # the rows are written as f32, so the entropy matches the Python floats to f32
        # precision rather than exactly
        assert abs(mean(report["pi_entropy"]) - expected_entropy) < 1e-6, report["pi_entropy"]
        assert print_report(report, skipped, len(selected), 1, board) is False

        # one colour winning everything is the degenerate case this command exists for
        black_dir = os.path.join(temp, "black_only")
        os.makedirs(black_dir)
        for game_id in range(0, 6 * 16, 16):
            write_game(os.path.join(black_dir, f"data_{game_id}_beefbeef"), 1, black_win)
        black_games, black_skipped = read_window(
            select_files(black_dir, ("",), None), board
        )
        black_report = audit(black_games)
        assert black_report["winners"] == {1: 6, -1: 0, None: 0}, black_report["winners"]
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
        assert print_report(abort_report, abort_skipped, 6, 1, board) is True

        # a window too small to judge must say so instead of crying degeneracy
        one_dir = os.path.join(temp, "one_game")
        os.makedirs(one_dir)
        write_game(os.path.join(one_dir, "data_0_11111111"), 1, white_win)
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

    games, skipped = read_window(selected, args.board)
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
