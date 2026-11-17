"""Offline build step: download the raw data and write data/tennis.duckdb.

Run once on your laptop (takes a few minutes):
    uv run build_data.py

Sources
  1. Jeff Sackmann, tennis_atp / tennis_wta (GitHub): every tour-level match, with seeds,
     entry codes (Q, WC, LL, PR) and rankings. Used for surface Elo and to rebuild each
     Slam's bracket.
  2. tennis-data.co.uk: closing betting odds per match. Used to check the Elo ratings.

Outputs (tables in data/tennis.duckdb)
  slams          one row per Slam draw (slam_id like '2024-WIM-ATP')
  draw           one row per player per Slam: slot, seed, rank, Elo, form, draw luck
  slam_matches   every Slam match with pre-match Elo probability and market probability
  model_checks   how well Elo predicts Slam matches vs. the betting market
  matches        all tour-level matches (for ad-hoc SQL)
  players        player bios
"""

import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

import sim

# --- Config ---

FIRST_ELO_YEAR = 2005  # Elo warm-up starts here
FIRST_SLAM_YEAR = 2015  # Slams we analyze
LAST_YEAR = date.today().year
TOURS = ["atp", "wta"]

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "tennis.duckdb"

# Sackmann's original repos stopped serving files, so we fall back to archive mirrors
# of the same CSVs (same license: CC BY-NC-SA 4.0). First URL that works wins.
SACKMANN_MIRRORS = [
    "https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{file}",
    "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main/{tour}/{file}",
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/{tour}/{file}",
]
TENNIS_DATA = [
    "https://www.tennis-data.co.uk/{folder}/{year}.xlsx",
    "http://www.tennis-data.co.uk/{folder}/{year}.xlsx",
]
# tennis-data.co.uk answers 403 to Python's default user agent
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

ROUND_ORDER = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}
SLAM_KEYS = {"australian open": "AO", "roland garros": "RG", "french open": "RG", "wimbledon": "WIM", "us open": "USO"}
SLAM_NAMES = {"AO": "Australian Open", "RG": "Roland Garros", "WIM": "Wimbledon", "USO": "US Open"}


# --- Download ---


def fetch(urls: list[str], dest: Path) -> str | None:
    """Download the first url that works to dest (skipped if dest exists).
    Returns the url used, or None if every url failed."""
    if dest.exists():
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
        except requests.RequestException:
            continue
        # Mirrors sometimes answer 200 with an HTML error page; real files never start with '<'
        if r.status_code == 200 and r.content and not r.content.lstrip()[:1] == b"<":
            dest.write_bytes(r.content)
            print(f"  {dest.name} ({len(r.content) // 1024} KB) from {url.split('/')[2]}")
            return url
    return None


def download():
    print("Downloading Sackmann match files...")
    for tour in TOURS:
        for file in [f"{tour}_players.csv"] + [f"{tour}_matches_{y}.csv" for y in range(FIRST_ELO_YEAR, LAST_YEAR + 1)]:
            if not fetch([u.format(tour=tour, file=file) for u in SACKMANN_MIRRORS], RAW / file):
                print(f"  {file}: not found on any mirror, skipping")

    print("Downloading tennis-data.co.uk odds...")
    for tour in TOURS:
        for year in range(FIRST_SLAM_YEAR, LAST_YEAR + 1):
            folder = str(year) if tour == "atp" else f"{year}w"
            urls = [u.format(folder=folder, year=year) for u in TENNIS_DATA]
            if not fetch(urls, RAW / f"odds_{tour}_{year}.xlsx"):
                print(f"  odds {tour} {year}: blocked or missing, skipping (download by hand, see README)")


# --- Load ---


def load_matches() -> pd.DataFrame:
    frames = []
    for tour in TOURS:
        for path in sorted(RAW.glob(f"{tour}_matches_*.csv")):
            if not re.fullmatch(rf"{tour}_matches_\d{{4}}\.csv", path.name):
                continue  # skip qualifying / futures files if any
            df = pd.read_csv(path, low_memory=False)
            df["tour"] = tour.upper()
            frames.append(df)
    if not frames:
        sys.exit("No match files in data/raw. Did the download fail?")
    m = pd.concat(frames, ignore_index=True)
    m["tourney_id"] = m["tour"] + "-" + m["tourney_id"].astype(str)  # ATP and WTA ids can collide
    m = m[m["round"].isin(ROUND_ORDER) | ~m["tourney_level"].eq("G")]
    m["tourney_date"] = pd.to_datetime(m["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    m["surface"] = m["surface"].fillna("Hard").replace({"Carpet": "Hard"})
    m["round_order"] = m["round"].map(ROUND_ORDER).fillna(-1)
    m["best_of"] = pd.to_numeric(m.get("best_of", 3), errors="coerce").fillna(3).astype(int)
    for side in ["winner", "loser"]:
        m[f"{side}_seed"] = pd.to_numeric(m[f"{side}_seed"], errors="coerce")
        m[f"{side}_rank"] = pd.to_numeric(m[f"{side}_rank"], errors="coerce")
        m[f"{side}_entry"] = m[f"{side}_entry"].fillna("").astype(str)
        m[f"{side}_id"] = m["tour"] + "-" + m[f"{side}_id"].astype(str)
    m = m.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"]).reset_index(drop=True)
    return m


def load_players() -> pd.DataFrame:
    frames = []
    for tour in TOURS:
        path = RAW / f"{tour}_players.csv"
        if path.exists():
            p = pd.read_csv(path, low_memory=False)
            p["player_id"] = tour.upper() + "-" + p["player_id"].astype(str)
            p["name"] = (p["name_first"].fillna("") + " " + p["name_last"].fillna("")).str.strip()
            p["tour"] = tour.upper()
            frames.append(p[["player_id", "name", "tour", "hand", "dob", "ioc"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- Elo ---


def k_factor(n_matches: int) -> float:
    return 250 / (n_matches + 5) ** 0.4  # FiveThirtyEight-style decaying K


def slam_key(name: str) -> str | None:
    n = name.lower()
    return next((v for k, v in SLAM_KEYS.items() if k in n), None)


def run_elo(m: pd.DataFrame):
    """One pass over every match in date order.

    Returns (match_elo, snapshots):
      match_elo: per-match pre-match blended Elo of winner and loser
      snapshots: {tourney_id: {player_id: dict of pre-tournament ratings and form}}
    """
    overall, n_overall = defaultdict(lambda: 1500.0), defaultdict(int)
    surf, n_surf = defaultdict(lambda: 1500.0), defaultdict(int)
    history = defaultdict(list)  # player -> [(date, won, overall_elo_after)]

    w_pre = np.zeros(len(m))
    l_pre = np.zeros(len(m))
    snapshots = {}

    def blend(p, s):
        return 0.5 * overall[p] + 0.5 * surf[(p, s)]

    for tid, t in m.groupby(["tourney_date", "tourney_id"], sort=False):
        first = t.iloc[0]
        if first["tourney_level"] == "G" and first["tourney_date"].year >= FIRST_SLAM_YEAR:
            start, s = first["tourney_date"], first["surface"]
            field = set(t["winner_id"]) | set(t["loser_id"])
            snap = {}
            for p in field:
                h = history[p]
                last_date = h[-1][0] if h else None
                recent = [x for x in h if x[0] >= start - timedelta(days=84)]
                six_months = [x for x in h if x[0] <= start - timedelta(days=182)]
                snap[p] = {
                    "elo": overall[p],
                    "elo_surface": surf[(p, s)],
                    "elo_blend": blend(p, s),
                    "career_matches": n_overall[p],
                    "days_since_last_match": (start - last_date).days if last_date is not None else None,
                    "matches_12w": len(recent),
                    "wins_12w": sum(x[1] for x in recent),
                    "elo_change_6m": overall[p] - (six_months[-1][2] if six_months else 1500.0),
                }
            snapshots[tid[1]] = snap

        for i, row in t.iterrows():
            w, l, s = row["winner_id"], row["loser_id"], row["surface"]
            w_pre[i], l_pre[i] = blend(w, s), blend(l, s)
            for ratings, counts, kw, kl in [(overall, n_overall, w, l), (surf, n_surf, (w, s), (l, s))]:
                exp_w = 1 / (1 + 10 ** ((ratings[kl] - ratings[kw]) / 400))
                ratings[kw] += k_factor(counts[kw]) * (1 - exp_w)
                ratings[kl] -= k_factor(counts[kl]) * (1 - exp_w)
                counts[kw] += 1
                counts[kl] += 1
            d = row["tourney_date"]
            history[w].append((d, 1, overall[w]))
            history[l].append((d, 0, overall[l]))

    m["winner_elo_pre"], m["loser_elo_pre"] = w_pre, l_pre
    return snapshots


# --- Brackets ---


def rebuild_bracket(t: pd.DataFrame) -> list[str] | None:
    """Rebuild the draw from results: start at the final and expand each player into
    the match they won in the round before. Returns player ids in slot order."""
    rounds = [r for r in ROUND_ORDER if r in set(t["round"])]
    final = t[t["round"] == "F"]
    if len(final) != 1:
        return None
    slots = [final.iloc[0]["winner_id"], final.iloc[0]["loser_id"]]
    for r in reversed(rounds[:-1]):
        by_winner = t[t["round"] == r].set_index("winner_id")
        nxt = []
        for p in slots:
            if p not in by_winner.index:
                return None
            row = by_winner.loc[p]
            if isinstance(row, pd.DataFrame):
                return None
            nxt += [p, row["loser_id"]]
        slots = nxt
    return slots


def orient(slots: list, seed_of: dict) -> list:
    """Flip halves so the better seed sits on top at every level (cosmetic only)."""
    if len(slots) == 1:
        return slots
    half = len(slots) // 2
    top, bottom = orient(slots[:half], seed_of), orient(slots[half:], seed_of)
    best = lambda xs: min((seed_of.get(p) or 999) for p in xs)  # noqa: E731
    return top + bottom if best(top) <= best(bottom) else bottom + top


def reached(t: pd.DataFrame, size: int) -> dict:
    """Actual round each player reached ('W' for the champion)."""
    names = sim.round_names(size)
    out = {}
    for _, row in t.iterrows():
        k = ROUND_ORDER[row["round"]] - (7 - int(np.log2(size)))
        out[row["loser_id"]] = names[k]
        if row["round"] == "F":
            out[row["winner_id"]] = "W"
    return out


def build_slams(m, snapshots, names):
    slam_rows, draw_rows, failed = [], [], []
    slams = m[(m["tourney_level"] == "G") & (m["tourney_date"].dt.year >= FIRST_SLAM_YEAR)]
    for tid, t in slams.groupby("tourney_id", sort=False):
        first = t.iloc[0]
        key = slam_key(first["tourney_name"])
        if key is None:
            continue
        year, tour = first["tourney_date"].year, first["tour"]
        slam_id = f"{year}-{key}-{tour}"
        slots = rebuild_bracket(t)
        if slots is None or len(slots) & (len(slots) - 1):
            failed.append(slam_id)
            continue
        size = len(slots)

        info = {}
        for side in ["winner", "loser"]:
            for _, r in t.iterrows():
                p = r[f"{side}_id"]
                info.setdefault(p, {"seed": r[f"{side}_seed"], "entry": r[f"{side}_entry"], "rank": r[f"{side}_rank"]})
        seed_of = {p: (int(v["seed"]) if pd.notna(v["seed"]) else None) for p, v in info.items()}
        slots = orient(slots, seed_of)

        # Player index = slot index in the real draw, so slots are just 0..size-1
        snap = snapshots[tid]
        elo = np.array([snap[p]["elo_blend"] for p in slots])
        seeds = np.array([seed_of[p] or 0 for p in slots])
        best_of = int(t["best_of"].max())
        real, rand = sim.draw_luck(elo, np.arange(size), seeds, best_of)
        rnames = sim.round_names(size)
        actual = reached(t, size)

        champion = t[t["round"] == "F"].iloc[0]["winner_id"]
        slam_rows.append({
            "slam_id": slam_id, "tour": tour, "year": year, "slam": SLAM_NAMES[key], "slam_key": key,
            "surface": first["surface"], "start_date": first["tourney_date"].date(), "draw_size": size,
            "best_of": best_of, "n_seeds": int((seeds > 0).sum()), "champion": names.get(champion, champion),
            "sackmann_tourney_id": tid,
        })
        elo_rank = (-elo).argsort().argsort() + 1
        for i, p in enumerate(slots):
            s = snap[p]
            row = {
                "slam_id": slam_id, "slot": i, "quarter": i // (size // 4) + 1, "eighth": i // (size // 8) + 1,
                "player_id": p, "player": names.get(p, p), "seed": seed_of[p], "entry": info[p]["entry"] or None,
                "rank": info[p]["rank"], "elo_rank_in_field": int(elo_rank[i]),
                **{k: s[k] for k in s},
                "actual_result": actual.get(p, rnames[0]),
            }
            for k, rn in enumerate(rnames[1:], start=1):
                row[f"p_{rn}_real"] = real[i, k]
                row[f"p_{rn}_random"] = rand[i, k]
            row["exp_wins_real"] = real[i, 1:].sum()
            row["exp_wins_random"] = rand[i, 1:].sum()
            row["luck_wins"] = row["exp_wins_real"] - row["exp_wins_random"]
            row["luck_title"] = real[i, -1] - rand[i, -1]
            draw_rows.append(row)
        print(f"  {slam_id}: {size} players, champion {names.get(champion, champion)}")
    if failed:
        print(f"  could not rebuild brackets for: {', '.join(failed)}")
    return pd.DataFrame(slam_rows), pd.DataFrame(draw_rows)


# --- Odds ---


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def td_surname(name: str) -> str:
    """'De Minaur A.' -> 'deminaur'; 'Auger-Aliassime F.' -> 'augeraliassime'."""
    parts = str(name).strip().split()
    while len(parts) > 1 and re.fullmatch(r"[A-Z]\.(-?[A-Z]\.)*", parts[-1]):
        parts.pop()
    return norm(" ".join(parts))


def load_odds() -> pd.DataFrame:
    td_round = {"1st Round": "R128", "2nd Round": "R64", "3rd Round": "R32", "4th Round": "R16",
                "Quarterfinals": "QF", "Semifinals": "SF", "The Final": "F"}
    frames = []
    for path in sorted(RAW.glob("odds_*.xlsx")):
        tour, year = path.stem.split("_")[1:]
        df = pd.read_excel(path)
        level_col = "Series" if "Series" in df.columns else "Tier"
        df = df[df[level_col].astype(str).str.contains("Grand Slam", na=False)].copy()
        if df.empty:
            continue
        w_odds = next((c for c in ["AvgW", "PSW", "B365W"] if c in df.columns), None)
        l_odds = w_odds[:-1] + "L" if w_odds else None
        if not w_odds:
            continue
        ow, ol = pd.to_numeric(df[w_odds], errors="coerce"), pd.to_numeric(df[l_odds], errors="coerce")
        df["p_market"] = (1 / ow) / (1 / ow + 1 / ol)
        df["tour"] = tour.upper()
        df["year"] = int(year)
        df["slam_key"] = df["Tournament"].map(lambda x: slam_key(str(x)))
        df["round"] = df["Round"].map(td_round)
        df["w_sur"] = df["Winner"].map(td_surname)
        df["l_sur"] = df["Loser"].map(td_surname)
        df["odds_source"] = f"tennis-data.co.uk {path.name} ({w_odds}/{l_odds})"
        frames.append(df[["tour", "year", "slam_key", "round", "w_sur", "l_sur", "p_market", "odds_source"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_slam_matches(m, slams, names, odds):
    keep = m[m["tourney_id"].isin(slams["sackmann_tourney_id"])].merge(
        slams[["sackmann_tourney_id", "slam_id", "slam_key", "year"]], left_on="tourney_id", right_on="sackmann_tourney_id"
    )
    out = pd.DataFrame({
        "slam_id": keep["slam_id"], "tour": keep["tour"], "year": keep["year"], "slam_key": keep["slam_key"],
        "round": keep["round"], "winner": keep["winner_id"].map(names), "loser": keep["loser_id"].map(names),
        "winner_seed": keep["winner_seed"], "loser_seed": keep["loser_seed"],
        "winner_rank": keep["winner_rank"], "loser_rank": keep["loser_rank"], "score": keep["score"],
        "best_of": keep["best_of"], "winner_elo_pre": keep["winner_elo_pre"], "loser_elo_pre": keep["loser_elo_pre"],
    })
    out["p_elo_winner"] = sim.win_prob(out["winner_elo_pre"].values, out["loser_elo_pre"].values, 3)
    out["p_elo_winner_bo5adj"] = np.where(out["best_of"] == 5, sim.bo5_from_bo3(out["p_elo_winner"]), out["p_elo_winner"])
    out["p_market_winner"] = np.nan
    out["odds_source"] = None
    if not odds.empty:
        # tennis-data names are 'Surname I.'; match on the surname being the end of
        # Sackmann's full name, within the same tour, year, Slam and round.
        by_round = defaultdict(list)
        for r in odds.itertuples():
            by_round[(r.tour, r.year, r.slam_key, r.round)].append(r)
        for i, r in out.iterrows():
            w, l = norm(r["winner"]), norm(r["loser"])
            for o in by_round.get((r["tour"], r["year"], r["slam_key"], r["round"]), []):
                if o.w_sur and o.l_sur and w.endswith(o.w_sur) and l.endswith(o.l_sur):
                    out.at[i, "p_market_winner"], out.at[i, "odds_source"] = o.p_market, o.odds_source
                    break
    return out


def model_checks(sm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tour, g in sm.groupby("tour"):
        both = g.dropna(subset=["p_market_winner"])
        for label, col, data in [
            ("Elo (best-of-3 formula), all Slam matches", "p_elo_winner", g),
            ("Elo (best-of-5 adjusted), all Slam matches", "p_elo_winner_bo5adj", g),
            ("Elo (best-of-5 adjusted), matches with odds", "p_elo_winner_bo5adj", both),
            ("Betting market (closing odds), matches with odds", "p_market_winner", both),
        ]:
            if data.empty:
                continue
            p = data[col].clip(1e-6, 1 - 1e-6)
            rows.append({
                "tour": tour, "model": label, "n_matches": len(data),
                "accuracy": float((p > 0.5).mean()),
                "brier": float(((1 - p) ** 2).mean()),
                "log_loss": float(-np.log(p).mean()),
            })
    return pd.DataFrame(rows)


# --- Main ---


def main():
    if "--no-download" not in sys.argv:
        download()
    print("Loading matches...")
    m = load_matches()
    players = load_players()
    names = dict(zip(players["player_id"], players["name"])) if not players.empty else {}
    for side in ["winner", "loser"]:
        names.update({k: v for k, v in zip(m[f"{side}_id"], m[f"{side}_name"]) if k not in names})
    print(f"  {len(m):,} matches, {m['tourney_date'].min().date()} to {m['tourney_date'].max().date()}")

    print("Running surface Elo...")
    snapshots = run_elo(m)

    print("Rebuilding Slam draws and simulating draw luck (20,000 draws each)...")
    slams, draw = build_slams(m, snapshots, names)

    print("Joining betting odds...")
    odds = load_odds()
    sm = build_slam_matches(m, slams, names, odds)
    checks = model_checks(sm)
    print(f"  odds matched for {sm['p_market_winner'].notna().sum():,} of {len(sm):,} Slam matches")
    print(checks.to_string(index=False))

    matches = m[["tour", "tourney_id", "tourney_name", "tourney_level", "surface", "tourney_date", "round",
                 "winner_name", "loser_name", "winner_rank", "loser_rank", "winner_seed", "loser_seed",
                 "winner_entry", "loser_entry", "score", "best_of", "winner_elo_pre", "loser_elo_pre"]].copy()
    matches["tourney_date"] = matches["tourney_date"].dt.date

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(DB_PATH))
    for name, df in [("slams", slams), ("draw", draw), ("slam_matches", sm), ("model_checks", checks),
                     ("matches", matches), ("players", players)]:
        con.register("df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
    con.close()
    print(f"Wrote {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB): {len(slams)} Slams, {len(draw):,} draw rows")


if __name__ == "__main__":
    main()
