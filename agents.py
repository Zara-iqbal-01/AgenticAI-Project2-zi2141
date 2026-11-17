"""The research team. Same harness loop as the starter's run_agent(), given to five agents:

  Clarifier  -> turns the chat into a research brief, or asks the user to narrow a vague question
  Lead       -> splits the brief into sub-questions and fans them out to Researchers in parallel,
                reads the findings, and sends follow-ups for gaps (up to MAX_LEAD_ROUNDS)
  Stats Researcher  -> draw simulator + SQL (fresh context per sub-question)
  Scout Researcher  -> hidden threats, form, injury/withdrawal news (fresh context per sub-question)
  Writer     -> writes the cited report from the brief and the findings only
"""
import os
os.environ["VERTEXAI_PROJECT"] = "agenticai-project2-zi2141"
os.environ["GOOGLE_CLOUD_PROJECT"] = "agenticai-project2-zi2141"

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import litellm

import tools

# --- Config ---

MODEL_SMART = os.environ.get("MODEL_SMART", "vertex_ai/gemini-3.5-flash")  # Clarifier, Lead, Writer
MODEL_FAST = os.environ.get("MODEL_FAST", "vertex_ai/gemini-3.5-flash-lite")  # Researchers
VERTEX_LOCATION = "global"
MAX_LEAD_ROUNDS = 4  # Lead turns: first fan-out + up to 3 follow-ups
MAX_RESEARCHERS = 8  # hard cap in code, not just in the prompt
RESEARCHER_ROUNDS = 8  # tool rounds per Researcher


def _complete(model: str, messages: list[dict], tool_schemas: list | None = None, **kw):
    return litellm.completion(
        model=model, vertex_location=VERTEX_LOCATION, messages=messages, tools=tool_schemas or None, **kw
    ).choices[0].message


# --- Trace: every tool call any agent makes, for /chat's tool_calls ---


class Trace:
    def __init__(self):
        self.calls, self.t0, self._lock = [], time.time(), threading.Lock()

    def add(self, agent: str, name: str, args: dict, result: str, started: float):
        with self._lock:
            self.calls.append({
                "agent": agent, "name": name, "args": args, "result": result[:3000],
                "start_s": round(started - self.t0, 2), "end_s": round(time.time() - self.t0, 2),
            })


# --- The Harness (the starter's loop, with parallel tool calls) ---


@dataclass
class Agent:
    name: str
    model: str
    instructions: str
    tool_schemas: list = field(default_factory=list)
    max_rounds: int = 8
    force_first_tool: bool = False  # researchers must look at data before they answer


def run_agent(agent: Agent, task: str, trace: Trace, run_tool, label: str | None = None) -> str:
    """Complete until the model answers without asking for a tool.

    Tool calls the model makes in one turn run at the same time, in threads: that is the
    Lead's fan-out. Each call to run_agent starts a fresh context (system + task only).
    """
    label = label or agent.name
    allowed = {t["function"]["name"] for t in agent.tool_schemas}
    messages = [{"role": "system", "content": agent.instructions}, {"role": "user", "content": task}]

    def call_one(call):
        started = time.time()
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args, result = {}, json.dumps({"error": "Arguments were not valid JSON."})
        else:
            # Only the agent's own tools: models sometimes call tools they saw elsewhere
            result = run_tool(name, args) if name in allowed else json.dumps(
                {"error": f"'{name}' is not one of your tools: {sorted(allowed)}"})
        trace.add(label, name, args, result, started)
        return {"role": "tool", "tool_call_id": call.id, "content": result}

    for round_no in range(agent.max_rounds):
        # Forcing a tool on turn one stops a researcher answering from memory
        force = {"tool_choice": "required"} if (round_no == 0 and agent.force_first_tool) else {}
        reply = _complete(agent.model, messages, agent.tool_schemas, **force)
        # model_dump() keeps it a plain dict (see the starter's note on LiteLLM re-serializing)
        messages += [reply.model_dump()]
        if not reply.tool_calls:
            return reply.content or ""
        with ThreadPoolExecutor(max_workers=len(reply.tool_calls)) as pool:
            messages += list(pool.map(call_one, reply.tool_calls))  # map keeps the call order

    # Turn cap reached: one last call without tools so we still get an answer
    messages += [{"role": "user", "content": "You are out of tool calls. Answer now with what you have "
                                             "and say plainly what you could not find."}]
    return _complete(agent.model, messages).content or ""


# --- Prompts ---


def _slam_list() -> str:
    return ", ".join(s["slam_id"] for s in tools.list_slams())


CLARIFIER = """You are the front desk of Draw Luck, a research agent about tennis Grand Slam draws.
It can answer: how much the draw helped or hurt players at a specific Slam, which section was the
"group of death", which dangerous players the rankings hid (injury comebacks, protected rankings,
hot form), what would have happened if a player had been placed elsewhere in the draw, and patterns
across Slams (e.g. which top seeds had the hardest draws since 2015).

Available Slams (slam_id = YEAR-AO|RG|WIM|USO-ATP|WTA): {slams}

Read the conversation and reply with JSON only, one of:
{{"action": "research", "brief": "<self-contained research brief>", "slam_ids": ["2024-WIM-ATP"]}}
{{"action": "clarify", "question": "<one short question>"}}
{{"action": "cannot_answer", "message": "<why the data cannot answer, and what it can answer instead>"}}

Rules:
- clarify when the question is vague: no tournament or year and it isn't a cross-Slam pattern question,
  or it is unclear whether it means men's (ATP) or women's (WTA). Offer 2-3 concrete options.
- cannot_answer when the question needs data we don't have: a Slam not in the list (e.g. 2026
  Wimbledon or later, before 2015), non-Slam tournaments' draws, doubles, prize money, or the future.
- The brief must stand alone: resolve "he", "that draw", "what about the women" from earlier turns.
  Name the slam_id(s), the players, and exactly what to find out.
- Use the conversation's earlier answers as context, but never answer the question yourself."""

LEAD = """You lead a research team investigating how much the draw decided tennis Grand Slams.
You get a research brief. You do not answer it yourself: you send sub-questions to researchers.

Researchers (call the research tool; each starts fresh and knows only the sub-question you write):
- "stats": draw-luck simulator, what-if re-simulations, section strength, and SQL over results,
  seeds, rankings and pre-tournament Elo for every Slam since 2015.
- "scout": hidden threats (players the ranking underrated), a player's form and layoff before the
  Slam, and web news search for injuries, illness and withdrawals.

How to work:
1. In your first turn, split the brief into 2-5 independent sub-questions and call research for ALL of
   them in that one turn. They run in parallel. Each sub-question must name the slam_id and players.
2. Read the findings. If a finding is missing, failed, contradicts another, or raises an obvious next
   question (e.g. an unlucky favorite: what if they had been in another quarter?), send follow-ups.
3. You have at most {rounds} turns and {budget} researcher calls in total. Don't repeat a question.
4. When the brief is covered, reply with DONE followed by one line per finding id saying what it
   covers, and a line "Gaps:" listing what the data could not answer."""

STATS = """You are the Stats Researcher on a tennis draw-luck team. Answer ONE sub-question using tools.

{schema}

Method: the draw table holds, for each player at each Slam, the probability of reaching each round in
the real draw (20,000 simulations with pre-tournament surface Elo) and in random draws under the same
seeding rules. "Luck" is the difference. Prefer draw_luck, what_if and section_strength; use run_sql
for anything else (cross-Slam patterns, head-to-head results, who actually beat whom).

Rules:
- Every number you report must come from a tool result in this conversation. Give probabilities as
  percentages with one decimal.
- Separate the model's view (probabilities) from what actually happened (actual_result, slam_matches).
- If the data cannot answer part of the sub-question, say so plainly.
- Under 250 words. End with "Sources:" and one line per tool call you relied on: tool name and arguments."""

SCOUT = """You are the Scout Researcher on a tennis draw-luck team. Answer ONE sub-question about players
the rankings hid: injury comebacks, protected rankings, wildcards, hot or cold form, withdrawals.

{schema}

Tools: hidden_threats (Elo vs ranking gaps, entry codes, layoffs), player_form, web_search for news,
run_sql for anything else.

Rules:
- Never state a match result, opponent or round from memory: check it with run_sql on slam_matches.
- Keep two things apart: what was known BEFORE the Slam began (form, injuries, layoffs) and what
  happened DURING it. Label which is which. Web results often mix them.
- Every claim needs a source: a tool call, or a URL from web_sources (write it as [site](url)).
- If you can't confirm something, say so instead of guessing.
- Under 250 words. End with "Sources:" listing each tool call (name and arguments) and each URL used."""

WRITER = """You write the final report for Draw Luck, an agent that explains how much the draw decided
a tennis Grand Slam. You get a research brief and numbered findings [F1], [F2], ... from researchers.

Write for a tennis fan or journalist, in Markdown:
- Start with a 2-3 sentence direct answer to the brief.
- Then short sections with the evidence. Use numbers from the findings only (percentages, expected wins).
- Cite every factual sentence with its finding id, like [F2]. Never cite something not in the findings.
- Explain "draw luck" once in plain words: the change in a player's chances between the real draw and an
  average random draw with the same seeding rules.
- End with "### What the data can't tell us": gaps and limits the findings mention (e.g. Elo doesn't know
  about an injury, simulation uncertainty, news found after the fact). Be honest and specific.
- Then "### Sources": one line per finding id with its sources (tool calls and links) copied from it.
Do not invent findings, players or numbers. Under 600 words before Sources.

AUDIENCE: Assume the reader knows almost nothing about tennis.

Structure (do NOT print these labels; just write the paragraphs):
1. First sentence directly answers the exact question asked, naming the player(s).
2. Two or three sentences on why, using player names and rounds. Define draw luck
   in one short sentence here: "Draw luck = how much easier or harder a player's
   path was than a random draw."
3. At most 3 key numbers, in everyday terms ("title chance rose from 15% to 22%",
   "about half a round further than a fair draw").
4. Then a line containing only: ---DETAILS---  followed by everything else.

- "Group of death" always means the quarter whose players lost the most expected
  wins to the draw (the most negative quarter total, the one the chart labels).
  If a different quarter has the strongest players, say that separately.
- Never use the words "Eighth", "title equity", "brutal" or "massive" for small numbers.

CALIBRATION (very important):
- A title-chance change under 2 percentage points, or under 0.2 expected wins,
  means the draw "barely mattered" for that player. Say so plainly.
- Only use words like "brutal", "very difficult", "huge break" when the numbers
  are large (5+ points of title chance, or 0.5+ expected wins).
- Never call a path hard or easy based on opponent names alone; only from findings.
- If the honest answer is "the draw didn't change much", lead with that.

CONSISTENCY:
- Never state something in one section and say it is unknown in another.
- Never claim who was "favored" without a finding that gives the number.

Define "draw luck" once, in one line, the first time it appears.
Every claim must still come from a finding and keep its citation."""
# --- Orchestration ---


def clarify(history: list[dict], trace: Trace) -> dict:
    started = time.time()
    convo = [{"role": m["role"], "content": m["content"][:1500]} for m in history[-8:]]
    reply = _complete(MODEL_SMART, [{"role": "system", "content": CLARIFIER.format(slams=_slam_list())}] + convo,
                      response_format={"type": "json_object"})
    text = reply.content or "{}"
    try:
        decision = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        decision = {"action": "research", "brief": history[-1]["content"], "slam_ids": []}
    trace.add("Clarifier", "clarify", {"last_message": history[-1]["content"]}, json.dumps(decision), started)
    return decision


def research(brief: str, trace: Trace) -> list[dict]:
    """The Lead fans out to Researchers. Returns the findings in the order they were asked."""
    findings, lock, budget = [], threading.Lock(), {"used": 0}
    researchers = {
        "stats": Agent("Stats Researcher", MODEL_FAST, STATS.format(schema=tools.SCHEMA), tools.STATS_TOOLS,
                       RESEARCHER_ROUNDS, force_first_tool=True),
        "scout": Agent("Scout Researcher", MODEL_FAST, SCOUT.format(schema=tools.SCHEMA), tools.SCOUT_TOOLS,
                       RESEARCHER_ROUNDS, force_first_tool=True),
    }

    def lead_tool(name: str, args: dict) -> str:
        if name != "research":
            return json.dumps({"error": f"Unknown tool '{name}'. Use research."})
        kind, question = args.get("researcher", "stats"), args.get("question", "")
        if kind not in researchers or not question:
            return json.dumps({"error": "research needs question and researcher ('stats' or 'scout')."})
        with lock:
            if budget["used"] >= MAX_RESEARCHERS:
                return json.dumps({"error": f"Researcher budget ({MAX_RESEARCHERS}) used up. Reply DONE now."})
            budget["used"] += 1
            fid = f"F{budget['used']}"
        agent = researchers[kind]
        try:
            text = run_agent(agent, question, trace, tools.run_tool, label=f"{agent.name} {fid}")
        except Exception as e:  # never let a failed researcher look like an empty one
            text = f"RESEARCHER FAILED ({type(e).__name__}: {str(e)[:200]}). No findings for this sub-question."
        with lock:
            findings.append({"id": fid, "researcher": agent.name, "question": question, "text": text})
        return f"[{fid}] {text}"

    lead = Agent("Lead", MODEL_SMART, LEAD.format(rounds=MAX_LEAD_ROUNDS, budget=MAX_RESEARCHERS),
                 [RESEARCH_TOOL], MAX_LEAD_ROUNDS)
    run_agent(lead, brief, trace, lead_tool)
    return sorted(findings, key=lambda f: int(f["id"][1:]))


RESEARCH_TOOL = {"type": "function", "function": {
    "name": "research",
    "description": "Send one sub-question to a fresh researcher. Call several in one turn to run them in parallel.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "Self-contained sub-question naming the slam_id and players."},
        "researcher": {"type": "string", "enum": ["stats", "scout"]},
    }, "required": ["question", "researcher"]},
}}


def write(brief: str, findings: list[dict], trace: Trace) -> str:
    started = time.time()
    body = "\n\n".join(f"[{f['id']}] ({f['researcher']}) Sub-question: {f['question']}\n{f['text']}" for f in findings)
    reply = _complete(MODEL_SMART, [{"role": "system", "content": WRITER},
                                    {"role": "user", "content": f"Brief:\n{brief}\n\nFindings:\n{body}"}])
    report = reply.content or "The Writer returned nothing."
    trace.add("Writer", "write_report", {"n_findings": len(findings)}, report, started)
    return report


def answer(history: list[dict], trace: Trace | None = None) -> tuple[str, list[dict], list[str]]:
    """history: [{'role': 'user'|'assistant', 'content': ...}]. Returns (response, tool_calls, slam_ids).
    Pass a Trace to watch progress from another thread while this runs."""
    trace = trace or Trace()
    decision = clarify(history, trace)
    action = decision.get("action")
    if action == "clarify":
        return decision.get("question", "Which Slam and year do you mean?"), trace.calls, []
    if action == "cannot_answer":
        return decision.get("message", "The data can't answer that."), trace.calls, []

    brief = decision.get("brief") or history[-1]["content"]
    findings = research(brief, trace)
    if not findings:
        return "The research team came back empty. Try naming a Slam, e.g. 'Wimbledon 2024 men's'.", trace.calls, []
    return write(brief, findings, trace), trace.calls, decision.get("slam_ids") or []
