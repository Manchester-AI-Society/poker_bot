# PocketAces submission

Prepared 7 November 2026 for the 18:00 Europe/London deadline.

## Submission files and form

All three required files are at the root of the `codex/pocketaces-bot` branch:

- `pocketaces.pkl` — the fresh, untrained Model instance, created with the organiser's submission tool (pickle protocol 4).
- `model.py` — strategy, event tracking, range sampling and opponent statistics.
- `poker_eval.py` — original pure-Python evaluator and range features.

`pocketaces-submission.zip` contains exactly these three files. Keep the source files and pickle together: the pickle names the class; the source supplies its implementation. `test_bot.py`, this report and `validation-results.txt` are for review and are not required uploads.

The requested name is **PocketAces**. The organiser's CLI permits only lowercase filenames, so the file is `pocketaces.pkl`.

**The live form differs from the README's upload wording.** When inspected on 7 November, it asked for a GitHub repository link, not file attachments. It has these four required fields:

1. Open [Poker Bot Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSe1MQUfkZcGnGBpcgON4J7HumKMBsOekWnr0ZvLcytxA8R5gQ/viewform?usp=dialog).
2. Enter `PocketAces` in **Team Name**.
3. Enter your actual team members in **Team Member's Names**.
4. Enter the leader's email in **Email (Please supply the leader's email)**.
5. In **Please add a link to your github repository**, paste this branch URL, which contains the pickle and both source files:
   `https://github.com/Advaith-Ganesh/poker_bot/tree/codex/pocketaces-bot`
6. Check that the linked branch shows all three files, then click **Submit** yourself and retain the form's confirmation. If the organiser adds file-upload fields later, upload the three individual files above after extracting the ZIP.

The form has not been submitted by the coding assistant.

## Implementation

The bot targets the current README and arena defaults: no-limit Hold'em, 500 chips, 10/20 blinds (25 big blinds), heads-up duplicate deals with stacks reset per hand, ranked by bb/100. It also plays legally at larger tables and with short or asymmetric stacks. The README and current arena event code take precedence over the stale undecided-format notes in CLAUDE.md. The arena worker currently allows 10 seconds per callback.

Preflop decisions use position, hand percentile (pairs, ranks, suitedness and connectivity), raise history and effective depth. Premium hands can reraise or commit at shallow depths; weak pairs do not automatically jam. Postflop decisions use sampled equity against nonuniform ranges conditioned on the opponent's betting history, plus pot odds, equity-realisation and sampling margins, board texture, position and stack-to-pot ratio. Controlled randomisation supplies occasional bluffs and semi-bluffs.

Ranges retain a weak-hand/bluff floor rather than assuming all aggression represents a premium hand. Opponent participation, raises, large preflop bets, postflop aggression and fold-to-bet responses are smoothed toward conservative priors. Large changes require at least 12 observed hands. Statistics are keyed by team name, never a permanent seat. Showdown cards are not confused with board cards; missing showdowns are not treated as evidence of a known holding.

Each callback consumes only its delivered event delta. A public ledger reconstructs before-action pot, price and remaining chips, then reconciles to the authoritative view. Seed resets hand history and caches while retaining learned statistics. All final strategy actions are validated; calculation failures or budget exhaustion return a legal check/fold fallback. Raises use integer street totals clamped to the actual legal range, respecting closed raising rights.

Simulation samples opponent holes and runouts without replacement, excluding hero and board cards, and splits ties correctly. Normal targets are 480 heads-up samples, 240 multiway samples, or 800 on the river. Preparation and simulation share a 0.65-second decision budget with periodic checks; exhausted range construction abandons the decision rather than using a biased truncated range. Caches reset each hand and are bounded to 16 entries. Their keys include the opponent, board, full stored action history, learned profile and table size; the hero's fixed hole cards are implicit within that hand.

Tables are constructed in `__init__`. Initial pickle globals are only `model.Model` and allowed `random.Random`. After play, the organiser's allowlisted `card_utils.Card` also appears. Attributes hold serialisable data and the explicitly permitted random generator. There is no runtime file access, network access, external executable, paid API, LLM call or additional dependency. The organiser's pinned Pydantic version is preserved. All new strategy and evaluator code is original; existing repository attribution and files remain intact.

## Reproduce

Python 3.12 was used for development; source syntax supports Python 3.10+.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/test_model.py -q
.venv/bin/python -m pytest test_bot.py -q
.venv/bin/python -m pytest -q
.venv/bin/python -m evals
.venv/bin/python -m evals --opponents tight equity --deals 1000 --table
.venv/bin/python -m submit --team pocketaces
```

The final submit command writes the pickle, scans its allowed globals and sources, stages a clean copy containing only the required files and plays it against every dummy. It reported `OK: played legally against every dummy`, with no vet findings or refused globals.

Changed/added repository files: `model.py`, `poker_eval.py`, `test_bot.py`, `pocketaces.pkl`, `pocketaces-submission.zip`, `BOT_REPORT.md`, `validation-results.txt`. Engine files, evaluation opponents, requirements and the existing tests are unchanged.

## Validation evidence

- Created the virtual environment and installed the exact development requirements.
- Initial mandated submission test: 9 passed. Final combined submission and added bot tests: 27 passed in 14.51 seconds.
- Full repository suite: 415 passed, 5 skipped in 128.00 seconds. The final budget guard was subsequently covered by the combined 27-test run.
- Additional tests cover raise bounds, maximum all-in totals, free checks, short-stack calls, closed raising rights following a short all-in, deadline and calculation fallbacks, street changes, incremental event consumption, swapped seats, team statistics, nonuniform ranges, card exclusion, shared runouts, tie splitting at 2/3/6/9 players, and pickling after real play.
- The new evaluator matched the organiser's evaluator on 36,000 random 5/6/7-card hands, plus 3,600 additional random hands and explicit wheel, flush, quads, two-trips/full-house and three-pair cases in the added tests. The reference seven-card evaluator measured 127.1 microseconds per hand; the fast evaluator measured approximately 3–4 microseconds.
- A separate 120-hand scenario run used 2/3/6/9 players, swapped seats/buttons and stacks from 7 to 2,000 chips. It recorded **0 faults, 0 internal fallbacks**, and a slowest decision of **0.07517 seconds**. Its learned pickle passed actual organiser vetting without findings.
- A synthetic stress case with 9 players and 256 stored betting observations deliberately exhausted the budget and returned a legal check in **0.6605 seconds**, with one expected internal fallback. This was not an engine fault and is distinct from normal match timing.
- The working first version was preserved before changes. No training project or opponent modifications were used.

## Evaluation results

All values are bb/100 ± the organiser's 95% confidence half-width. Seed 0, one repeat, default 500-chip stacks and 10/20 blinds. Each heads-up deck is replayed with seats swapped; 500 deals therefore means 1,000 played hands. Intervals use the arena's per-deck duplicate scoring.

Default `.venv/bin/python -m evals` (500 deals per opponent):

| Opponent | PocketAces bb/100 | Played hands | Faults | Slowest callback |
|---|---:|---:|---:|---:|
| fold | +63.6 ± 2.4 | 1,000 | 0 | 0.04 s |
| call | +283.7 ± 40.4 | 1,000 | 0 | 0.17 s |
| random | +124.6 ± 44.6 | 1,000 | 0 | 0.14 s |
| shove | +161.6 ± 78.1 | 1,000 | 0 | 0.04 s |
| tight | +38.5 ± 12.8 | 1,000 | 0 | 0.13 s |
| equity | +179.8 ± 35.0 | 1,000 | 0 | 0.12 s |

The untouched repository template (always call/check) was preserved and evaluated with the **same** settings as the requested longer command: `--opponents tight equity --deals 1000 --table`, seed 0, one repeat. This is a comparison against that template, not against a solver or a prior competitive bot.

| Match | Template bb/100 | PocketAces bb/100 | Hands per bot/version |
|---|---:|---:|---:|
| heads-up tight | −94.7 ± 24.1 | +43.0 ± 8.7 | 2,000 |
| heads-up equity | −253.7 ± 27.9 | +163.4 ± 24.4 | 2,000 |
| table with tight and equity | −321.5 ± 32.7 | +82.6 ± 19.0 | 3,000 |

All longer matches, including the three-player table, had 0 faults. The slowest PocketAces callback was 0.23 seconds heads-up and 0.14 seconds at the table. The baseline also had 0 faults. The exact requested command completed successfully.

These results support improvement over the always-call template on these opponents. They do not establish optimal strategy or predict the tournament winner. The bot uses heuristic ranges and a limited number of simulations, not a solved strategy; small samples and approximate range assumptions affect marginal decisions. Multiway joint ranges and side-pot valuation are approximate. The statistical intervals describe deck variation for these fixed opponent/seed settings and do not cover unknown opposing strategies. The deadline guard may deliberately check/fold in an unusually expensive state.

## Submission checksums (SHA-256)

```text
e7106ca6f7fc31bb391f7e7c5349a6370ebc8f0a7d9d983dd8135c7ab1274ef4  pocketaces.pkl
85af07655d8274f753a625d9cb936947e51078b406a4bcfdb0c372e0f9a947e5  model.py
78a59bb3311d769355e18f578d3482ce773a22577035047a978107b57bbd59ec  poker_eval.py
```
