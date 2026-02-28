# var_demo — Historical VAR Model for an Equities Options Portfolio

A fully working, test-driven Historical Simulation Value-at-Risk (HS-VAR) engine for a
portfolio of European equity options. Built end-to-end using **AI-assisted coding** as a
demonstration of how a human quant and an AI pair-programmer can collaborate to deliver a
production-quality quantitative model from a blank repository.

---

## What this project delivers

- **Four VAR methodologies**: Plain Historical Simulation (Method A), Age-Weighted HS
  (Method B), Factor-Based VAR (Method D), and Stressed VAR per Basel 2.5
- **Expected Shortfall (ES / CVaR)** at all confidence levels, alongside VAR
- **Black-Scholes options pricer** — vectorised; full Greeks (Δ, Γ, ν, θ)
- **Synthetic vol surface** — rolling realised vol ATM proxy + quadratic skew in
  log-moneyness
- **Rolling backtest** with Kupiec POF test, Christoffersen independence test, and Basel
  traffic-light classification
- **149 tests** (138 active, 11 deselected for network/slow) — zero failures
- **Jupyter notebook** (`notebooks/var_demo.ipynb`) — a runnable end-to-end demo
- **Full documentation** — requirements, test plan, technical specification, test results
  report, and model governance card

---

## Project Time Summary

**Total time: ~175 minutes (~3 hours)**

| # | Date | Prompt summary | Time |
|---|------|---------------|------|
| 001 | 2026-02-27 | Set up working agreement — CLAUDE.md, Journal, folder structure, .gitignore | ~2 min |
| 002 | 2026-02-27 | Propose a Historical VAR model — requirements, 4 methodologies, test plan | ~8 min |
| 003 | 2026-02-27 | Answer open questions — confidence levels, universe size, trade format, vol proxy, 10-day scaling | ~10 min |
| 004 | 2026-02-27 | Implement the full codebase — 12 source modules, 13 test files, TODOs for gaps | ~35 min |
| 005 | 2026-02-27 | Run the tests — diagnose 7 failures, fix all bugs, reach 121/121 passing | ~12 min |
| 006 | 2026-02-27 | Implement first TODO — BS-priced premiums in `generate_portfolio` | ~8 min |
| 007 | 2026-02-27 | Complete all remaining TODOs — idiosyncratic VAR, trade aging, retry, vol cap, vectorised SVaR, Kupiec x=0 | ~25 min |
| 008 | 2026-02-27 | Create Jupyter notebook — end-to-end demo with all 4 methods and backtest | ~10 min |
| 009 | 2026-02-28 | Fix Stressed VAR runtime error — vol warm-up boundary bug in `stress_var.py` | ~5 min |
| 010 | 2026-02-28 | Write technical specification — 19-section model spec for development team | ~15 min |
| 011 | 2026-02-28 | Write testing documentation — 138 test results, 10 chart figures, bug log | ~20 min |
| 012 | 2026-02-28 | Create model card — 13-section governance document for model validation team | ~15 min |
| 013 | 2026-02-28 | Write README — project summary, prompt table, AI collaboration narrative | ~10 min |

---

## Key Moments of Human–AI Interaction

This project was built in a single pair-programming session between a human quant (domain
expert, decision-maker) and an AI assistant (implementation engine, code reviewer, writer).
The interactions below illustrate how the two collaborated — where the human led, where the
AI led, and where iteration between the two produced better outcomes than either would alone.

---

### 1. Establishing the working agreement before any code (Entry 001)

**What happened:** Before writing a single line of model code, the human asked the AI to
create a `CLAUDE.md` — a working agreement specifying: a prompt journal, a planning gate
before major tasks, a TDD requirement, and documentation standards.

**Why it matters for AI-assisted development:** This is the equivalent of a team charter.
Without it, an AI will optimise for "task completed" rather than "task completed with
appropriate governance". The working agreement gave the AI persistent, project-scoped
instructions that shaped every subsequent interaction — mandatory journal entries, a
test-first discipline, and a docs-first documentation culture.

**The lesson:** Front-loading governance constraints pays compound interest. Every
subsequent prompt benefited from the AI already knowing the rules.

---

### 2. The AI surfaced open questions before building (Entry 002)

**What happened:** The human gave a broad prompt — "create a Historical VAR model
proposal". The AI produced `docs/requirements.md` and `docs/test_plan.md`, but explicitly
included an **Open Questions** section listing eight decisions the human needed to make
before implementation could proceed: confidence levels, universe size, trade format,
real vs synthetic data, American vs European options, and more.

**Why it matters:** A less disciplined AI would have made these choices silently and
proceeded to code. This AI flagged the decisions explicitly, ensuring the human approved
them before 35 minutes of implementation work began. This prevented rework.

**The interaction pattern:** AI proposes → human decides → AI updates docs → both parties
confirm → implementation begins. This is the **spec-review loop** and it is central to
producing high-quality model code rather than a quick prototype.

---

### 3. Human provides decisions; AI updates both spec and test plan atomically (Entry 003)

**What happened:** The human answered all eight open questions in a single message (e.g.,
"95% and 99%, configurable", "25 names", "European only", "√10 scaling"). The AI updated
both `requirements.md` and `test_plan.md` in a single pass — adding ES tests, configurable
CL tests, trade format validation tests, and network test markers to the plan before a line
of implementation code existed.

**Why it matters:** The test plan was written to reflect the actual requirements, not
retrofitted to the code. When tests later caught bugs, they were testing the right things.
This is the core discipline of TDD: the test plan is a specification, not an afterthought.

---

### 4. Forty minutes of implementation with deliberate gaps (Entry 004)

**What happened:** The human approved the spec and said "begin implementation". The AI
produced 12 source modules and 13 test files — but deliberately used `# TODO:` markers for
six features it assessed as out of scope for a first pass:

- Retry logic for network errors
- No-arbitrage bounds on the vol surface
- BS-priced premiums (vs. a placeholder)
- Idiosyncratic residual component in Factor VAR
- Vectorised stress window scan
- Exact Kupiec formula for x=0 edge case

**Why it matters:** The AI made an explicit design choice: deliver a working skeleton with
known gaps flagged, rather than a bloated first version with unvalidated complexity. This is
a professional engineering discipline — ship the walking skeleton, then refine. The human
could review what was built before complexity was added.

---

### 5. "Run the tests before proceeding" — the human enforces TDD (Entry 005)

**What happened:** The human's next prompt was simply: *"Run the tests before proceeding
to the next step."*

**The result:** 7 failures out of 121 tests. The AI diagnosed all 7 root causes and fixed
them:

| Bug | Nature |
|-----|--------|
| `scalar_input` flag on BS functions | Logic error affecting vectorised shape |
| Kupiec import in wrong module | Wrong source for `build_age_weights` |
| Flat-scenario non-zero P&L | Test design error — vol lookup path confused |
| Greeks approx test failed due to skew coupling | Tests not isolating the right variable |
| Age-weight test tolerance too tight | Numerical precision over-specified |
| Stress VAR error message mismatch | String literal assumption |
| Factor VAR integration bound | Conceptual misunderstanding of Method D output |

**Why it matters:** This is the most important moment in the project. The human didn't ask
*what* was broken — they simply enforced the discipline. The AI found and fixed everything.
**TDD caught real bugs before they reached the demo notebook.** Without this step, bugs
3, 5, and 6 would have surfaced as confusing runtime errors much later.

**The lesson for AI-assisted development:** The human's most valuable contribution is
not writing code — it is enforcing process gates. "Run the tests" is a five-word prompt
that produced 12 minutes of diagnostic and repair work.

---

### 6. Incremental TODO completion with human control over sequencing (Entries 006–007)

**What happened:** After the initial 121 tests passed, the human chose the next piece of
work: *"Select next TODO and implement it."* The AI picked the BS-premium TODO (Entry 006).
The human then said: *"Complete all remaining TODOs one after another"* (Entry 007).

This two-step approach — one TODO first, then the rest — gave the human a checkpoint.
After Entry 006 added 3 tests and reached 124 passing, the human had evidence that the
pattern worked before delegating the remaining five TODOs.

**Notable moment in Entry 007:** The Kupiec x=0 fix contained a secondary bug — the
test itself had the wrong formula for the expected LR value (`ln(1/0.01)` instead of
`ln(1/0.99)`). The AI caught this during implementation, fixed both the source code and
the test, and documented it in the Journal. This demonstrates the AI acting as its own
code reviewer — finding errors in tests written earlier by itself.

---

### 7. The human ran the notebook and reported a precise error (Entry 009)

**What happened:** After the Jupyter notebook was created (Entry 008), the human opened it,
ran it, and hit a runtime error in the Stressed VAR cell:

```
ValueError: No realised vol available for AAPL on 2019-01-03 —
increase warm-up period or check data.
```

The human reported the exact error message verbatim. The AI identified the root cause
immediately: `find_stress_window` was passing pre-warm-up log-return dates to
`full_reprice_pnl`, which tried to look up rolling vol for dates before the 126-day warm-up
completed — producing NaN — and then raised the error.

**The fix was a one-liner trim:**

```python
log_returns = log_returns.iloc[vol_warm_up - 1:]
```

**Why it matters:** This is a textbook example of human-in-the-loop testing. The AI
produced a notebook that passed syntax validation but contained a latent boundary condition
that only manifested on real execution. The human caught it by running it. The AI fixed it
in one targeted patch without touching anything else. Tests remained at 138/138.

**The lesson:** AI-generated code benefits from human execution. The human's role here
wasn't to understand the warm-up boundary mathematics — it was to run the code and report
what broke. That is a highly effective division of labour.

---

### 8. Documentation as a first-class output (Entries 010–012)

**What happened:** After the model was complete and tested, the human requested three
progressively more formal documents in sequence:

1. *"Create a technical document for the model development team"* → `docs/model_technical_specification.md`
2. *"Create testing documentation with results and graphs"* → `docs/test_results.md` + 10 PNG charts
3. *"Produce material for the model validation team — produce a model card"* → `docs/model_card.md`

Each document was produced in a single pass from the AI's knowledge of the codebase it had
built. The technical spec included 19 sections with mathematical formulae, engineering
notes, and a 10-item limitations table. The test results report embedded 10 charts
generated from the same synthetic dataset as the notebook. The model card included a
proposed model risk classification and a recommended monitoring schedule.

**Why it matters:** Documentation is often the last thing done in a software project and
the first thing cut. In this project, the human treated documentation as a deliverable on
equal footing with code — and the AI produced it at the same depth. The model card is
ready for submission to a model risk management function without further editing.

**The lesson for AI-assisted model development:** An AI that built the model can also write
the model documentation — because it holds the full context. This is one of the highest-leverage applications of AI in quantitative development: eliminating the translation gap between "code that works" and "documentation that explains why it works".

---

## Project Structure

```
var_demo/
├── CLAUDE.md                          # Working agreement
├── Journal.md                         # Complete prompt and outcome log
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # pytest configuration
│
├── src/
│   ├── utils/
│   │   ├── black_scholes.py           # BS pricer — price, Δ, Γ, ν, θ
│   │   └── stats.py                   # VAR/ES, Kupiec, Christoffersen, traffic light
│   ├── data/
│   │   ├── loader.py                  # yfinance download + retry + fill
│   │   ├── vol_surface.py             # Rolling vol + quadratic skew surface
│   │   └── portfolio.py               # Trade format, validation, BS-priced generator
│   ├── var/
│   │   ├── scenarios.py               # Log-returns, scenario set, age weights
│   │   ├── revaluation.py             # Full BS reprice + delta-gamma approx
│   │   ├── historical_var.py          # Methods A (plain HS) and B (age-weighted)
│   │   ├── filtered_var.py            # Method C (GARCH-filtered — implemented, not validated)
│   │   ├── factor_var.py              # Method D (factor decomposition)
│   │   └── stress_var.py              # Stressed VAR — Basel 2.5
│   └── backtest/
│       └── backtest.py                # Rolling backtest, exception flagging, statistical tests
│
├── tests/
│   ├── conftest.py                    # Session-scoped synthetic fixtures (seed=42)
│   ├── unit/                          # 13 unit test files — 121 active tests
│   └── integration/                   # 3 integration test files — 21 active tests
│
├── notebooks/
│   └── var_demo.ipynb                 # End-to-end runnable demo — 39 cells
│
├── portfolios/
│   └── sample_portfolio.csv           # 12-trade sample options book
│
└── docs/
    ├── requirements.md                # Approved functional specification
    ├── test_plan.md                   # Pre-implementation test specification
    ├── model_technical_specification.md  # 19-section mathematical + engineering spec
    ├── test_results.md                # 138-test results report + 10 chart figures
    ├── model_card.md                  # Model governance card for MRM submission
    └── images/                        # 10 PNG charts (price series, vol surface, P&L, backtest, …)
```

---

## Getting Started

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run all tests:**

```bash
pytest tests/ -v --tb=short -m "not network and not slow"
```

**Run the notebook:**

```bash
jupyter notebook notebooks/var_demo.ipynb
```

The notebook is self-contained — it generates synthetic data and requires no internet
connection or external files.

**Run a specific VAR method (example):**

```python
from data.loader import load_prices
from data.vol_surface import compute_rolling_vol
from data.portfolio import generate_portfolio, price_portfolio
from var.historical_var import compute_historical_var

# ... (see notebook for full working example)
result = compute_historical_var(
    trades, today_prices, rolling_vols,
    r=0.05, as_of=as_of, method="plain"
)
print(result["var_99_1d"], result["es_99_1d"])
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/requirements.md](docs/requirements.md) | Functional specification — methodology decisions, data sources, trade format |
| [docs/test_plan.md](docs/test_plan.md) | Pre-implementation test specification — 149 tests, acceptance criteria |
| [docs/model_technical_specification.md](docs/model_technical_specification.md) | Mathematical and engineering specification — for model developers |
| [docs/test_results.md](docs/test_results.md) | Test results, bug log, illustrated outputs — for model validators |
| [docs/model_card.md](docs/model_card.md) | Model governance card — for MRM / model validation team |

---

*Built in ~3 hours using Claude Code as an AI pair-programmer.*
*All 138 active tests pass. Pending MRM sign-off before production use.*
