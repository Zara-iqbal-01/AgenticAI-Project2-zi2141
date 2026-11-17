"""The tools the researchers can run, and the JSON that describes them to the model.

Same shape as the starter: plain Python functions that return JSON strings, a list of
tool schemas, and run_tool() that never lets a bad call crash the loop. Every result
carries a "source" so the Writer can cite it.
"""

import difflib
import json
import os
import re
from pathlib import Path

import duckdb
import numpy as np

import sim

DB_PATH = Path(__file__).parent / "data" / "tennis.duckdb"
SEARCH_MODEL = os.environ.get("SEARCH_MODEL", "gemini-3.5-flash")
MAX_ROWS = 60


def _query(sql: str, params: list | None = None) -> list[dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, _clean(r))) for r in cur.fetchall()]
    finally:
        con.close()


def _clean(row):
    """Round floats and make dates JSON-safe so results stay short."""
    out = []
    for v in row:
        if isinstance(v, float):
            v = None if np.isnan(v) else round(v, 4)
        elif hasattr(v, "isoformat"):
            v = v.isoformat()
        out.append(v)
    return out


def _dump(obj) -> str:
    return json.dumps(obj, default=str)


def list_slams() -> list[dict]:
    return _query("SELECT slam_id, tour, year, slam, surface, champion FROM slams ORDER BY start_date, tour")


def _slam(slam_id: str) -> dict | None:
    rows = _query("SELECT * FROM slams WHERE slam_id = ?", [slam_id])
    return rows[0] if rows else None


def _field(slam_id: str) -> list[dict]:
    return _query("SELECT * FROM draw WHERE slam_id = ? ORDER BY slot", [slam_id])


def _find(field: list[dict], name: str) -> dict | None:
    """Match a player name loosely: 'sinner', 'J. Sinner', 'Jannik Sinner'."""
    key = re.sub(r"[^a-z ]", "", name.lower()).strip()
    for p in field:
        if p["player"].lower() == key:
            return p
    for p in field:
        if key and key.split()[-1] in p["player"].lower().split():
            return p
    close = difflib.get_close_matches(name, [p["player"] for p in field], n=1, cutoff=0.6)
    return next((p for p in field if close and p["player"] == close[0]), None)


def _bad_slam(slam_id: str) -> str:
    return _dump({"error": f"Unknown slam_id '{slam_id}'. Format is YEAR-AO|RG|WIM|USO-ATP|WTA, "
                           f"e.g. '2024-WIM-ATP'. Use run_sql on the slams table to list them."})


# --- Tools ---


def run_sql(query: str) -> str:
    """Read-only SQL over the tennis database."""
    if not re.match(r"^\s*(select|with)\b", query, re.I):
        return _dump({"error": "Only SELECT or WITH queries are allowed."})
    try:
        rows = _query(query)
    except duckdb.Error as e:
        return _dump({"error": f"SQL error: {e}"})
    return _dump({"source": "tennis.duckdb (SQL)", "query": query, "n_rows": len(rows),
                  "rows": rows[:MAX_ROWS], "truncated": len(rows) > MAX_ROWS})


def draw_luck(slam_id: str, sort: str = "unluckiest", limit: int = 10, player: str | None = None) -> str:
    """How much each player's draw helped or hurt them, vs. the average random draw."""
    slam = _slam(slam_id)
    if not slam:
        return _bad_slam(slam_id)
    field = _field(slam_id)
    cols = ["player", "seed", "rank", "elo_rank_in_field", "quarter", "exp_wins_real", "exp_wins_random",
            "luck_wins", "p_QF_real", "p_QF_random", "p_W_real", "p_W_random", "luck_title", "actual_result"]
    if player:
        p = _find(field, player)
        if not p:
            return _dump({"error": f"No player like '{player}' in {slam_id}."})
        rows = [p]
    else:
        key = {"unluckiest": lambda r: r["luck_wins"], "luckiest": lambda r: -r["luck_wins"],
               "favorites": lambda r: -r["p_W_real"], "title_luck": lambda r: r["luck_title"]}.get(sort)
        if key is None:
            return _dump({"error": "sort must be unluckiest, luckiest, favorites or title_luck"})
        rows = sorted(field, key=key)[: max(1, min(int(limit), 30))]
    return _dump({
        "source": f"draw table, slam_id={slam_id} ({sim.N_SIMS:,} simulations of the real draw vs. "
                  f"{sim.N_SIMS:,} random draws under the same seeding rules)",
        "slam": f"{slam['year']} {slam['slam']} {slam['tour']}, champion {slam['champion']}",
        "how_to_read": "luck_wins = expected match wins in the real draw minus in a random draw. "
                       "Negative = the draw cost them wins. p_* are probabilities of reaching that round.",
        "players": [{c: r.get(c) for c in cols} for r in rows],
    })


def what_if(slam_id: str, player: str, target: str) -> str:
    """Re-simulate the draw with a player moved: swap with another player, or into quarter Q1-Q4."""
    slam = _slam(slam_id)
    if not slam:
        return _bad_slam(slam_id)
    field = _field(slam_id)
    p = _find(field, player)
    if not p:
        return _dump({"error": f"No player like '{player}' in {slam_id}."})
    elo = np.array([r["elo_blend"] for r in field])
    size, bo, i = len(field), slam["best_of"], p["slot"]
    names = sim.round_names(size)
    base = sim.simulate(elo, np.arange(size), bo)

    def summary(reach, k):
        return {"exp_wins": round(float(reach[k, 1:].sum()), 2),
                **{f"p_{names[r]}": round(float(reach[k, r]), 4) for r in range(len(names) - 4, len(names))}}

    q = re.fullmatch(r"q([1-4])", target.strip().lower())
    if q:
        quarter = int(q.group(1))
        cands = [r["slot"] for r in field if r["quarter"] == quarter and not r["seed"] and r["slot"] != i]
        if not cands:
            return _dump({"error": f"No unseeded spots in quarter {quarter}."})
        runs = []
        for j in cands:
            slots = np.arange(size)
            slots[i], slots[j] = j, i
            runs.append(sim.simulate(elo, slots, bo, n=3000, seed=j))
        moved = np.mean(runs, axis=0)
        top = sorted([r for r in field if r["quarter"] == quarter and r["slot"] != i], key=lambda r: -r["elo_blend"])[:4]
        return _dump({
            "source": f"what_if simulation on draw table, slam_id={slam_id}: {p['player']} placed in each of "
                      f"{len(cands)} unseeded spots of Q{quarter}, 3,000 simulations each",
            "player": p["player"], "from_quarter": p["quarter"], "to_quarter": quarter,
            "actual_draw": summary(base, i), "moved": summary(moved, i),
            "strongest_in_new_quarter": [{"player": r["player"], "seed": r["seed"], "elo": round(r["elo_blend"])} for r in top],
        })

    other = _find(field, target)
    if not other:
        return _dump({"error": f"target must be Q1-Q4 or a player in {slam_id}; no player like '{target}'."})
    j = other["slot"]
    slots = np.arange(size)
    slots[i], slots[j] = j, i
    swapped = sim.simulate(elo, slots, bo)
    return _dump({
        "source": f"what_if simulation on draw table, slam_id={slam_id}: swapped draw positions of "
                  f"{p['player']} and {other['player']}, {sim.N_SIMS:,} simulations",
        p["player"]: {"actual_draw": summary(base, i), "after_swap": summary(swapped, i)},
        other["player"]: {"actual_draw": summary(base, j), "after_swap": summary(swapped, j)},
    })


def section_strength(slam_id: str, level: str = "quarter") -> str:
    """How stacked each section of the draw was: find the 'group of death'."""
    if not _slam(slam_id):
        return _bad_slam(slam_id)
    col = "eighth" if level == "eighth" else "quarter"
    field = _field(slam_id)
    out = []
    for s in sorted({r[col] for r in field}):
        sec = sorted([r for r in field if r[col] == s], key=lambda r: -r["elo_blend"])
        out.append({
            col: s,
            "top_players": [f"{r['player']} (seed {r['seed'] or '-'}, Elo {round(r['elo_blend'])})" for r in sec[:4]],
            "mean_elo_top4": round(float(np.mean([r["elo_blend"] for r in sec[:4]]))),
            "title_equity_random_draw": round(sum(r["p_W_random"] for r in sec), 3),
            "total_luck_wins": round(sum(r["luck_wins"] for r in sec), 2),
            "reached_semis_from_here": [r["player"] for r in sec if r["actual_result"] in ("F", "W") or r["actual_result"] == "SF"],
        })
    return _dump({
        "source": f"draw table, slam_id={slam_id}, grouped by {col}",
        "how_to_read": "title_equity_random_draw = summed title chances of the section's players in a random "
                       "draw (pure strength). total_luck_wins < 0 means the section's players lost expected "
                       "wins to the draw, i.e. they were packed together.",
        "sections": sorted(out, key=lambda r: -r["mean_elo_top4"]),
    })


def hidden_threats(slam_id: str, limit: int = 8) -> str:
    """Players the ranking underrated: Elo far above ranking, protected ranking, or back from a layoff."""
    if not _slam(slam_id):
        return _bad_slam(slam_id)
    rows = _query("""
        WITH f AS (
          SELECT *, rank() OVER (ORDER BY coalesce(rank, 9999)) AS rank_in_field
          FROM draw WHERE slam_id = ?)
        SELECT player, seed, rank, entry, rank_in_field, elo_rank_in_field,
               rank_in_field - elo_rank_in_field AS ranking_gap, round(elo_blend) AS elo,
               days_since_last_match, matches_12w, wins_12w, round(elo_change_6m) AS elo_change_6m,
               career_matches, actual_result
        FROM f
        WHERE (seed IS NULL OR seed > 8) AND elo_rank_in_field <= 40
          AND (rank_in_field - elo_rank_in_field >= 8 OR entry IN ('PR','WC','Q','LL') OR days_since_last_match > 60)
        ORDER BY elo_rank_in_field LIMIT ?""", [slam_id, int(limit)])
    return _dump({
        "source": f"draw table, slam_id={slam_id} (pre-tournament Elo, ranking, entry code, form)",
        "how_to_read": "ranking_gap = how many places better their Elo ranks them in this field than their ATP/WTA "
                       "ranking does. Entry PR = protected ranking (back from injury), WC = wildcard, Q = qualifier. "
                       "Elo does not decay during a layoff, so a returning player keeps their old strength.",
        "players": rows,
    })


def player_form(slam_id: str, player: str) -> str:
    """One player's pre-tournament profile and last 10 matches before this Slam."""
    slam = _slam(slam_id)
    if not slam:
        return _bad_slam(slam_id)
    p = _find(_field(slam_id), player)
    if not p:
        return _dump({"error": f"No player like '{player}' in {slam_id}."})
    recent = _query("""
        SELECT tourney_date, tourney_name, surface, round,
               CASE WHEN winner_name = ? THEN 'W' ELSE 'L' END AS result,
               CASE WHEN winner_name = ? THEN loser_name ELSE winner_name END AS opponent, score
        FROM matches
        WHERE (winner_name = ? OR loser_name = ?) AND tourney_date < ? AND tour = ?
        ORDER BY tourney_date DESC LIMIT 10""",
        [p["player"]] * 4 + [slam["start_date"], slam["tour"]])
    keep = ["player", "seed", "rank", "entry", "elo", "elo_surface", "elo_blend", "elo_rank_in_field",
            "days_since_last_match", "matches_12w", "wins_12w", "elo_change_6m", "actual_result"]
    return _dump({"source": f"draw + matches tables, slam_id={slam_id}",
                  "profile": {k: p[k] for k in keep}, "last_10_matches_before": recent})


_genai_client = None


def web_search(query: str) -> str:
    """Google Search through Gemini grounding. Returns a summary plus the source URLs."""
    global _genai_client
    try:
        from google import genai
        from google.genai import types

        if _genai_client is None:
            import google.auth

            _, project = google.auth.default()
            _genai_client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT", project),
                                         location="global")
        resp = _genai_client.models.generate_content(
            model=SEARCH_MODEL,
            contents=f"Search the web and report the facts that answer: {query}\n"
                     "Be specific about dates. Say clearly if you find nothing.",
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
        )
        meta = resp.candidates[0].grounding_metadata
        sources = [{"title": c.web.title, "url": c.web.uri} for c in (meta.grounding_chunks or []) if c.web] if meta else []
        return _dump({"source": "Google Search via Gemini grounding", "summary": resp.text, "web_sources": sources[:8]})
    except Exception as e:  # the model can't see an exception; give it something to reason about
        return _dump({"error": f"Web search failed: {type(e).__name__}: {str(e)[:200]}"})


# --- What the model sees ---


def _fn(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


SLAM_ID = {"type": "string", "description": "e.g. '2024-WIM-ATP'. Format YEAR-AO|RG|WIM|USO-ATP|WTA."}
PLAYER = {"type": "string", "description": "Player name, e.g. 'Jannik Sinner' or 'Sinner'."}

T_SQL = _fn("run_sql", "Run a read-only DuckDB SELECT over the tables described in your instructions.",
            {"query": {"type": "string"}}, ["query"])
T_LUCK = _fn("draw_luck", "Draw luck for a Slam: how much each player's real draw helped or hurt them vs. a random "
             "draw with the same seeding rules. Use player= for one player, or sort for a ranked list.",
             {"slam_id": SLAM_ID, "sort": {"type": "string", "enum": ["unluckiest", "luckiest", "favorites", "title_luck"]},
              "limit": {"type": "integer"}, "player": PLAYER}, ["slam_id"])
T_WHATIF = _fn("what_if", "Counterfactual: re-simulate the Slam with a player moved. target is another player's "
               "name (swap their draw positions) or a quarter 'Q1'-'Q4' (averaged over its unseeded spots).",
               {"slam_id": SLAM_ID, "player": PLAYER, "target": {"type": "string"}}, ["slam_id", "player", "target"])
T_SECTION = _fn("section_strength", "Rank the quarters (or eighths) of a draw by strength to find the 'group of death'.",
                {"slam_id": SLAM_ID, "level": {"type": "string", "enum": ["quarter", "eighth"]}}, ["slam_id"])
T_HIDDEN = _fn("hidden_threats", "Unseeded or low-seeded players the ranking underrated: big Elo-vs-ranking gap, "
               "protected ranking or wildcard entry, or a long layoff before the Slam.",
               {"slam_id": SLAM_ID, "limit": {"type": "integer"}}, ["slam_id"])
T_FORM = _fn("player_form", "A player's pre-tournament ratings, layoff, recent record and last 10 matches before the Slam.",
             {"slam_id": SLAM_ID, "player": PLAYER}, ["slam_id", "player"])
T_WEB = _fn("web_search", "Search the web (news) for injuries, withdrawals, illness or form around a tournament. "
            "Include the player, tournament and year in the query.",
            {"query": {"type": "string"}}, ["query"])

STATS_TOOLS = [T_SQL, T_LUCK, T_WHATIF, T_SECTION]
SCOUT_TOOLS = [T_HIDDEN, T_FORM, T_WEB, T_SQL]

TOOL_MAP = {"run_sql": run_sql, "draw_luck": draw_luck, "what_if": what_if, "section_strength": section_strength,
            "hidden_threats": hidden_threats, "player_form": player_form, "web_search": web_search}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return _dump({"error": f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}"})
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return _dump({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:
        return _dump({"error": f"{name} failed: {type(e).__name__}: {str(e)[:200]}"})


SCHEMA = """Tables in tennis.duckdb (DuckDB SQL):
- slams(slam_id, tour, year, slam, slam_key, surface, start_date, draw_size, best_of, n_seeds, champion)
- draw(slam_id, slot, quarter, eighth, player, seed, entry, rank, elo, elo_surface, elo_blend, elo_rank_in_field,
       career_matches, days_since_last_match, matches_12w, wins_12w, elo_change_6m, actual_result,
       p_R64_real, p_R32_real, p_R16_real, p_QF_real, p_SF_real, p_F_real, p_W_real (and *_random),
       exp_wins_real, exp_wins_random, luck_wins, luck_title)
  One row per player per Slam. Ratings are as of the day before the Slam.
  THREE Elo columns, don't mix them: elo = all surfaces; elo_surface = this surface only (starts at 1500,
  so it is low for players with few matches on it); elo_blend = 50/50 mix, the rating the simulator uses.
  When you quote "Elo" for a player, quote elo_blend and call it "Elo". actual_result is the round they lost in
  ('R128' = lost first round, 'W' = champion). entry: Q, WC, LL, PR (protected ranking) or NULL.
- slam_matches(slam_id, tour, year, slam_key, round, winner, loser, winner_seed, loser_seed, winner_rank, loser_rank,
       score, best_of, winner_elo_pre, loser_elo_pre, p_elo_winner, p_market_winner, odds_source)
  *_elo_pre = blended Elo right before THAT match (it moves during the tournament, so it differs a little
  from draw.elo_blend). p_elo_winner = the model's pre-match probability that the eventual winner won.
- model_checks(tour, model, n_matches, accuracy, brier, log_loss): how well Elo predicts Slam matches
- matches(tour, tourney_id, tourney_name, tourney_level, surface, tourney_date, round, winner_name, loser_name,
       winner_rank, loser_rank, winner_seed, loser_seed, winner_entry, loser_entry, score, best_of,
       winner_elo_pre, loser_elo_pre): every tour-level match 2005 onward
Data covers Slams from 2015 to 2026 Roland Garros (no 2020 Wimbledon; it was cancelled)."""
