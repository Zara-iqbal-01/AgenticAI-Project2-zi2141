# Draw Luck 🎾
**How much did the draw decide a tennis Grand Slam?**

Live app: https://agenticai-project2-zi2141-1003268432355.us-central1.run.app
Path: **Deep Research** · Author: Zara Iqbal (zi2141)

## What it does and who it's for

Every Grand Slam starts with a random draw, and that draw quietly shapes who wins. Draw Luck is a research team of agents that measures how much. Ask it about any men's or women's Grand Slam from 2015 to 2026 Roland Garros and it will tell you:

- whose chances the draw raised or lowered (**draw luck**)
- which quarter was the **group of death**
- which **dangerous players** the rankings hid
- **what-if** swaps ("What if Alcaraz had been in Djokovic's spot?")

It's for **tennis fans, journalists and analysts** who want to know whether a favorite was robbed by the draw, backed by numbers rather than commentary. Answers start in plain language so people who don't follow tennis closely can understand them, with the full numbers one click away.

**Draw luck** = a player's expected match wins in their real draw minus their expected wins in a fair, random draw made under the same seeding rules.

## Data

| Source | What it gives | How it's loaded |
|---|---|---|
| Jeff Sackmann's ATP/WTA match data (CC BY-NC-SA 4.0), via the [Hugging Face archive mirror](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive) because the original GitHub repos stopped serving files | Every tour-level match since 2005: players, seeds, rounds, scores | `build_data.py` downloads it |
| [tennis-data.co.uk](http://www.tennis-data.co.uk) betting odds | Pre-match odds for every Slam match | `build_data.py` downloads it |
| Gemini with Google Search (Vertex AI) | News context: injuries, withdrawals, form | `web_search` tool, live |

`build_data.py` turns the raw files into `data/tennis.duckdb` (committed, so the deploy ships with it):

1. Computes a **surface-blended Elo rating** for every player from every match since 2005.
2. Rebuilds the bracket for all **90 Grand Slam draws from 2015 to 2026 Roland Garros**. Every rebuilt champion matches the real one.
3. Plays each real draw **20,000 times**, then 20,000 random draws that follow the real seeding rules. The difference is each player's draw luck.
4. Checks the ratings against the betting market (Brier score). A best-of-5 adjustment for men's Slams made predictions *worse* (0.1833 vs 0.1817), so it is switched off.

To rebuild from scratch: `uv run build_data.py` (a few minutes).

None of this exists anywhere online. Gemini knows who won each Slam, but it cannot know how 20,000 simulated versions of each draw played out.

## How the agents work

```
Question → Clarifier → Lead → Researchers (in parallel) → Lead (follow-ups) → Writer → cited answer
```

| Agent | Model | Job |
|---|---|---|
| **Clarifier** | gemini-3.5-flash | Asks a question back when the input is vague ("Who got unlucky?" → which Slam?) |
| **Lead** | gemini-3.5-flash | Plans the research and fans out Researchers in one turn. Reads their findings and sends follow-ups for gaps |
| **Stats / Scout Researchers** | gemini-3.5-flash-lite | Each runs in a fresh context with its own tools and returns one finding (F1, F2, …) |
| **Writer** | gemini-3.5-flash | Writes a plain-language answer where every claim cites a finding, then a "Show the numbers" section |

- **Parallel:** Researchers launched in the same Lead turn run at once (`ThreadPoolExecutor` in `agents.py`).
- **Goes back for more:** the Lead gets up to `MAX_LEAD_ROUNDS = 4` turns (first fan-out plus up to 3 follow-up rounds). Ask "Which quarter was the group of death at the 2025 US Open men's?" and the timeline shows the Lead returning for a second round.
- **Caps enforced in code, not just in prompts:** at most `MAX_RESEARCHERS = 8` per question, and each agent has a max number of turns.
- **Shows its work:** `/chat` keeps the Project 1 response shape. Every sub-agent tool call is in `tool_calls` with an `agent` field, and the front end draws them as the "How the team worked" timeline.

**Tools** (`tools.py`): `draw_luck`, `what_if`, `section_strength`, `hidden_threats`, `player_form`, `web_search`, `run_sql` (read-only).

## Front end

`index.html` was rebuilt from the starter:
- An intro panel saying what the app explores, who it's for and how to use it
- A **slot-by-slot draw chart** for any Slam, showing who the draw helped (blue) and hurt (red), with quarter totals and the group of death marked
- A live **agent timeline**: each bar is one tool call, and you can click it for details
- Clickable citations, and a short answer with the full numbers folded under "Show the numbers"

## Sample queries

Plain Gemini answers come from the same model (`gemini-3.5-flash`) with **no tools and no system prompt** (`no_tools.py`, full output in `no_tools_answers.txt`).

### 1. What if Alcaraz had been in Djokovic's spot at Wimbledon 2023?

**Expected answer:** Alcaraz's title chance rises from about **10.5% to 13.6%** (reaching the quarterfinals goes from 48.8% to 58.8%, expected wins from 3.70 to 4.02). Djokovic, moved into Alcaraz's spot, drops from **60.0% to 55.5%**. Djokovic's half was the easier one, and Alcaraz won the title from the harder half.
**Source:** `what_if(slam_id='2023-WIM-ATP', player='Alcaraz', target='Djokovic')` on the draw tables in `data/tennis.duckdb`.

**Draw Luck:**
> Swapping Carlos Alcaraz into Novak Djokovic's spot in the Wimbledon 2023 draw would have given Alcaraz an easier path with higher odds of reaching the final and winning the tournament. In his actual championship run, Alcaraz faced a congested section of the draw, encountering dangerous players like Matteo Berrettini in the round of 16 and top seeds like Holger Rune and Daniil Medvedev in the quarterfinals and semifinals. By contrast, Djokovic's spot offered a smoother path with fewer threats.
> - His pre-tournament title chance would have risen from 10.5% to 13.6%.
> - His probability of reaching the quarterfinals would have jumped from 48.8% to 58.8%.
> - His expected match wins would have climbed from 3.70 to 4.02.

**Plain Gemini (no tools):**
> *(excerpt)* "In 2023, Djokovic's draw was considered highly dangerous… If Alcaraz had been in Djokovic's spot at Wimbledon 2023, **Novak Djokovic likely wins his 24th Grand Slam**… The combination of a brutal physical path to the final (Hurkacz and Sinner)…"

**Difference:** Plain Gemini tells a confident story with no numbers and gets the direction wrong: it calls Djokovic's path the brutal one. The simulation shows Djokovic's side was the *easier* one.

### 2. Which quarter was the group of death at the 2025 US Open men's?

**Expected answer:** **Quarter 4** (top seed Taylor Fritz, with Djokovic). Its players lost a combined **1.40 expected wins** to the draw, the most of any quarter. Q2 gained the most (+1.05).
**Source:** `section_strength(slam_id='2025-USO-ATP')` / quarter totals of `draw_luck` in `data/tennis.duckdb` (also shown under the app's chart).

**Draw Luck:**
> Quarter 4 of the draw was considered the "group of death" at the 2025 US Open, threatening top players like Novak Djokovic and Taylor Fritz with a highly concentrated field of rivals. High-profile seeds like Holger Rune and dangerous unseeded players like Sebastian Korda made this quarter of the bracket particularly hazardous. The quarter featured an average rating of 1758.3, the highest of any quarter in the tournament. It also packed in 12 players who were either seeded in the top 16 or rated 1800 or higher.

**Plain Gemini (no tools):**
> "I cannot answer this question because the 2025 US Open has not taken place yet… the official draw… will not be released until just before the tournament begins."

**Difference:** The tournament happened. Plain Gemini's knowledge stops before it, while Draw Luck answers from the rebuilt bracket and 20,000 simulations.

### 3. Who had the easiest draw at the 2026 US Open men's?

**Expected answer:** The data can't answer this. The database covers Slams through **2026 Roland Garros**, and the 2026 US Open isn't in it. The agent should say so and offer the Slams it does cover.
**Source:** the `slams` table in `data/tennis.duckdb` (no `2026-USO-ATP` row).

**Draw Luck:**
> I do not have draw data for the 2026 US Open (my database for 2026 currently only goes up to Roland Garros). I can, however, analyze draw luck, groups of death, and player paths for any US Open from 2015 through 2025, or other completed Grand Slams in that period.

**Plain Gemini (no tools):**
> "I cannot answer this question because the 2026 US Open has not taken place yet, and the tournament draws have not been made."

**Difference:** Both decline, but only Draw Luck gives the right reason. The tournament did happen, and the agent checks its own data and says exactly where its coverage ends.

## What the data can't answer

- Slams after **2026 Roland Garros**, and draws before 2015
- Injuries, illness or fitness during a tournament. Ratings come from past results, though the `web_search` tool can add news context.
- Exact match-by-match pressure or momentum. The simulations give probabilities, not certainties.
- Qualifying rounds and lucky losers are taken as given, not simulated.

The Writer is told to say when a question falls outside the data, and to call small effects small.

## Run locally

```bash
uv sync
gcloud auth application-default login   # Vertex AI access
uv run app.py                           # then open http://localhost:8000
```

Use `localhost`, not `0.0.0.0`: the page needs a secure context.

## Deploy

Cloud Run with continuous deploy from this GitHub repo (Google Cloud buildpacks), region `us-central1`, 2 GiB memory, 600 s request timeout. Every push to `main` redeploys.