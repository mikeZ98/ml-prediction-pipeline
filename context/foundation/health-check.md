---
project: mlpp
checked_at: 2026-07-31T09:41:46Z
health_status: needs-attention
context_type: brownfield
language_family: python
stack_assessment_available: true
checks_run:
  - lockfile
  - dependency_audit
  - outdated_deps
  - test_runner
  - ci_cd
  - configuration
audit_findings:
  critical: 0
  high: 0
  moderate: 1
  low: 0
test_runner_detected: true
ci_provider: GitHub Actions
recommended_fixes: 7
---

# Health Check — mlpp

Audited 2026-07-31 against the project at `apps/backend/`, in the context of the `mlpp Dashboard`
change described in `context/foundation/prd.md`.

This report runs standalone rather than inside a teaching chain, and both a CI pipeline and agent
instruction files already exist — so findings are given as a single ranked list rather than split
into "fix now" and "covered later" buckets.

## Dependency Health

### Lockfile

```
Status: present (apps/backend/uv.lock, 344 KB, git-tracked)
Package manager: uv 0.11.7
```

Stronger than mere presence: `.github/workflows/ci.yml` syncs with `uv sync --locked`, which fails
the build if the lockfile has drifted from `pyproject.toml` rather than silently re-resolving.
Reproducibility is enforced, not assumed. No action needed.

### Security Audit

```
Tool: OSV database (api.osv.dev querybatch) against the fully-resolved locked set
      — pip-audit was attempted first and aborted (see note below)
Summary: 0 CRITICAL, 0 HIGH, 1 MODERATE, 0 LOW
Packages scanned: 159 (all groups: runtime + dev + notebook + dashboard)
Direct vs transitive: the single finding is a direct dev-group dependency
```

*Note on tooling:* `pip-audit` is not installed, and running it via `uvx` failed — it builds a
throwaway virtualenv for dependency resolution and `ensurepip` aborted with SIGABRT in this
environment, both with and without `--no-deps`. The audit was completed instead by exporting the
locked dependency set (`uv export --all-groups`) and querying OSV directly, which is the same
advisory source pip-audit consumes. Coverage is equivalent; only the transport differed.

#### MODERATE findings

- **pytest** 8.4.2 — `CVE-2025-71176` (aliases `GHSA-6w46-j5rx-g56g`, `PYSEC-2026-1845`):
  *"pytest through 9.0.2 on UNIX relies on directories with the `/tmp/pytest-of-{user}` name
  pattern, which allows local users to cause a denial of service or possibly gain privileges."*
  CVSS 3.1 `AV:L/AC:L/PR:N/UI:N/S:C/C:L/I:L/A:L`. Fixed in **9.0.3**.

  Two mitigating facts and one complication. It is a dev-only dependency with a local attack vector,
  so on a single-user machine the practical exposure is minimal. But the project pins
  `pytest>=8.3,<9`, which **excludes the fixed version** — so this cannot be resolved by a routine
  lockfile refresh. Clearing it requires widening the constraint, which also crosses a major version
  boundary. See fix #2.

### Outdated Dependencies

```
Packages with major version gaps: 3
```

- **mypy**: 1.20.2 → 2.3.0 (1 major behind)
- **pytest**: 8.4.2 → 9.1.1 (1 major behind — same constraint that blocks the advisory fix)
- **ipykernel**: 6.31.0 → 7.3.0 (1 major behind)

Minor gaps, informational only: `h5py` 3.14.0 → 3.16.0, `ipython` 9.15.0 → 9.16.0,
`jupyter-builder` 1.1.1 → 1.2.0, `keras` 3.15.0 → 3.15.1, `ruff` 0.15.22 → 0.16.1.

Nothing is dangerously stale. All three major gaps are development tooling, not runtime
dependencies — the ML stack (TensorFlow, numpy, pandas, scikit-learn) is current.

## Test Suite

```
Test runner: pytest 8.4.2 (+ pytest-cov)
Tests found: 145 across 10 modules
Test execution: passing (exit code 0, full suite)
Configuration: apps/backend/pyproject.toml [tool.pytest.ini_options]
```

This is the strongest area of the project, and it verified better than documented.

| Suite | Tests | Runtime | Notes |
|---|---|---|---|
| Fast (`-m 'not slow'`) | 98 | **1.56s** | `config`, `data`, `metrics`, `preprocess`, `session` |
| Slow (`-m slow`) | 47 | — | `model`, `notebook`, `pipeline`, `predict`, `predict_cli` |
| Full | 145 | **2m 15s** | all passing |

**The TensorFlow-free guarantee is real, and I verified it empirically rather than taking the
configuration's word for it.** After executing the entire fast suite in-process, neither
`tensorflow` nor `keras` appears in `sys.modules`. The invariant that `CLAUDE.md` documents — and
that PRD guardrail FR-019 depends on — holds in fact, not just in intent.

Supporting machinery worth noting: `--strict-markers` means a typo'd marker fails rather than
silently skipping tests, and `filterwarnings = ["error::DeprecationWarning:mlpp.*"]` turns the
project's own deprecation warnings into failures.

**One documentation drift.** `CLAUDE.md` states the full suite takes "~30s". It takes **2m 15s** on
this machine — roughly 4.5× the documented figure. The fast-suite claim ("~1s") is accurate. This
matters because an agent told a command takes 30 seconds may treat a 2-minute run as a hang.

## CI/CD

```
Provider: GitHub Actions
Configuration: .github/workflows/ci.yml
```

| Stage | Status | Notes |
|---|---|---|
| Lint | ✓ | `ruff check --output-format=github` |
| Format | ✓ | `ruff format --check` |
| Test | ✓ | full `pytest` suite (not just the fast subset) |
| Type check | ✓ | `mypy src/mlpp` |
| Build | ✗ | no wheel build step — but the package is never distributed, so this is not a real gap |
| Security | ✗ | **no dependency audit, no Dependabot, no CodeQL** |

Two jobs run on every push and pull request to `main`: a `quality` job across a Python 3.12/3.13
matrix, and a `notebook-drift` job that verifies the committed notebook still matches its generator.
Concurrency cancellation and least-privilege `permissions: contents: read` are both configured.

For a solo project this is a well-built pipeline — it covers four of the five stages that matter
here. The one genuine gap is security: nothing in CI would have surfaced the pytest advisory above,
and there is no `.github/dependabot.yml` to propose dependency updates.

## Configuration

### High severity

None.

### Medium severity

- **`.github/dependabot.yml`** — absent. No automated dependency update proposals, so advisories
  like the pytest one surface only when someone manually audits. Fix: see #3.

### Low severity

- **`.editorconfig`** — absent. `ruff format` governs Python, so the practical gap is limited to
  non-Python files (YAML, Markdown, the notebook JSON). Fix: see #6.
- **`apps/backend/src/mlpp/py.typed`** — absent (carried forward from the stack assessment). Only
  affects consumers importing the built wheel. Fix: see #7.
- **`.pre-commit-config.yaml`** — absent. Genuinely optional here: CI already enforces the same four
  checks, and a solo author running `ruff check` locally gets the same signal. Listed for
  completeness, not recommended.

Present and in good order: `.gitignore` (thorough — notably `OUTPUTS/*` with a `!OUTPUTS/example/`
negation, and project-local cache isolation matching `.envrc`), `.env.template`, `.envrc`,
`LICENSE`, `README.md`, `CLAUDE.md`, `.cursorrules`, `.cursorignore`.

## Stack Assessment Cross-Reference

```
Stack assessment: context/foundation/stack-assessment.md
Agent readiness (from stack-assess): ready-with-compensation
```

| Quality Gate Gap | Health-Check Finding | Status |
|---|---|---|
| Typed — pass (mypy strict + ruff ANN + CI) | `mypy src/mlpp` clean across 14 source files; CI enforces on both Python versions | **Reinforced** |
| Conventions — pass for package | 145 tests mirror source modules 1:1; `--strict-markers` and notebook-drift gate both active | **Reinforced** |
| Conventions — **fail** for Streamlit | No dashboard code exists yet, and no conventions written. Nothing here mitigates it | **Still open** |
| Keras 3 training-data skew (partial) | The full suite emits a Keras/TensorFlow `__array__` copy-keyword deprecation against numpy 2 — concrete evidence of version-boundary friction in this stack | **Reinforced** |
| uv training-data thinness (partial) | Lockfile committed and CI syncs `--locked`; `CLAUDE.md` already carries the "uv only" rule | **Mitigated** |
| Instruction files forbid the planned change | Both `CLAUDE.md` and `.cursorrules` are unchanged since the assessment — still contradicting the PRD | **Still open** |
| No `py.typed` marker | Confirmed absent | **Confirmed** |

## Recommended Fixes

Ranked by impact on agent-assisted work, not by generic severity.

### 1. A numpy deprecation sits directly on the artifact-loading path

**Impact**: This is the most consequential finding in the audit, and it was not visible from
configuration alone. `joblib` 1.5.3 sets `array.shape = ...` during unpickling, which **numpy 2.5.1
has deprecated**. It fires on `session.read_fitted_state()` — the function every `load_session()`
call goes through, and therefore the path every dashboard panel in the PRD depends on. I confirmed
it directly against the committed reference session:

```
$ uv run python -W error::DeprecationWarning -c "…read_fitted_state(Path('OUTPUTS/example'))"
DEPRECATION ON ARTIFACT LOAD: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
```

When numpy promotes this from deprecation to removal, `joblib.load` of every existing `scaler.gz`,
`output_scaler.gz` and `encoders.gz` breaks. That takes `OUTPUTS/example/` with it — which PRD
guardrail FR-021 requires to keep loading, and which the notebook report and CI drift gate both
read. The project's own warning filter does not catch this: `error::DeprecationWarning:mlpp.*`
matches only warnings originating in `mlpp` modules, and this one originates in `joblib`.

The exposure is live rather than hypothetical, because `pyproject.toml` pins `numpy>=2.0,<3` — the
version that removes this is inside the allowed range and will arrive on the next lockfile refresh.

**Severity**: high (latent — no current breakage) · **Effort**: moderate (15–30 min)

**Fix**: pick one.

```bash
# Option A — track upstream first. joblib may already have a fix in 1.6.x.
cd apps/backend && uv add 'joblib>=1.6,<2' && uv run pytest -m 'not slow'

# Option B — pin numpy below the removal until joblib is clear, and record why.
#   In pyproject.toml:  "numpy>=2.0,<2.6",   # joblib<1.6 unpickling trips a 2.5 deprecation

# Option C — make the failure loud instead of silent, so it cannot surprise you:
#   add to [tool.pytest.ini_options] filterwarnings:
#     "error::DeprecationWarning:joblib.*"
```

Option C is worth doing regardless of A or B — it converts a future silent break into a failing
test.

### 2. The pytest advisory's fix is outside the pinned range

**Impact**: `CVE-2025-71176` is MODERATE and dev-only, so the direct risk on a single-user machine
is low. The reason it ranks here is structural rather than severity-driven: the fix lands in pytest
9.0.3 while the project pins `>=8.3,<9`, so no routine lockfile refresh will ever clear it. It will
sit in every future audit until the constraint moves.

**Severity**: medium · **Effort**: quick (< 5 min to change; budget more if the major bump surfaces
test breakage)

**Fix**:

```bash
cd apps/backend
# widen the constraint in pyproject.toml:  "pytest>=9.0.3,<10"
uv lock && uv run pytest        # verify all 145 still pass under pytest 9
```

Worth pairing with the `mypy` 1.x → 2.x bump (#5) since both are dev-tooling major versions and
both want a full-suite verification run anyway.

### 3. CI has no security scanning

**Impact**: Four of five relevant CI stages are covered, but nothing in the pipeline checks
dependencies for advisories. The pytest finding above surfaced only because this audit ran by hand
— CI would not have caught it, and will not catch the next one.

**Severity**: medium · **Effort**: quick (< 5 min)

**Fix**: add `.github/dependabot.yml` —

```yaml
version: 2
updates:
  - package-ecosystem: "uv"
    directory: "/apps/backend"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
```

Optionally add an audit step to the `quality` job:

```yaml
      - name: Audit dependencies
        run: uv export --all-groups --no-hashes -o /tmp/reqs.txt && uvx pip-audit -r /tmp/reqs.txt
        continue-on-error: true
```

### 4. `CLAUDE.md` understates the full suite runtime by ~4.5×

**Impact**: `CLAUDE.md` documents the full suite as "~30s"; it measured **2m 15s**. An agent that
believes a command finishes in 30 seconds may treat a two-minute run as a hang and kill it, or
avoid running the full suite before declaring work done — which is precisely the check
`CLAUDE.md` asks for.

**Severity**: medium · **Effort**: quick (< 5 min)

**Fix**: in `CLAUDE.md` `## Testing patterns`, change the `uv run pytest` comment from
`# full suite incl. Keras smoke tests (~30s)` to `(~2min; TensorFlow import dominates)`. Confirm the
fast-suite "~1s" claim stays — it measured 1.56s and is accurate.

### 5. mypy is a major version behind

**Impact**: mypy 1.20.2 → 2.3.0. Type checking is this project's single strongest agent-legibility
asset, so keeping the checker current directly protects that. A major bump may surface new errors in
otherwise-unchanged code.

**Severity**: low-medium · **Effort**: moderate (15–30 min, depending on new diagnostics)

**Fix**:

```bash
cd apps/backend
# in pyproject.toml:  "mypy>=2.3,<3"
uv lock && uv run mypy src/mlpp
```

### 6. No `.editorconfig`

**Impact**: `ruff format` already governs every Python file, so the real gap is limited to YAML,
Markdown and JSON. Minor, but cheap to close.

**Severity**: low · **Effort**: quick (< 5 min)

**Fix**: create `.editorconfig` at the repository root —

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2

[*.py]
indent_size = 4
max_line_length = 100

[*.md]
trim_trailing_whitespace = false
```

### 7. No `py.typed` marker

**Impact**: `mlpp`'s types are invisible to anything installing it as a dependency. Irrelevant while
the package is only used locally; a one-line fix if that ever changes.

**Severity**: low · **Effort**: quick (< 5 min)

**Fix**:

```bash
touch apps/backend/src/mlpp/py.typed
# then in pyproject.toml:
#   [tool.hatch.build.targets.wheel]
#   packages = ["src/mlpp"]
#   force-include = { "src/mlpp/py.typed" = "mlpp/py.typed" }
```

### Carried forward from the stack assessment

Not repeated in detail here — see `context/foundation/stack-assessment.md` for paste-ready text:

- **`CLAUDE.md` and `.cursorrules` both currently forbid the planned dashboard.** Still unchanged.
  This remains the single highest-priority item before implementation begins, and it is an
  instruction-file fix rather than a health issue, which is why it lives in that report.
- **Streamlit conventions are unwritten.** No dashboard code exists yet, so nothing in this audit
  mitigates the gap.

## Summary

```
Health status: needs-attention
```

**This is a well-maintained project, and most of what I checked came back clean.** Dependencies are
locked and CI fails on lockfile drift; all 145 tests pass; lint, format and strict type checking are
clean across 14 source files and enforced in CI on two Python versions. The documented
TensorFlow-free split is not merely configured but genuinely true — I verified that running the
entire 98-test fast suite leaves neither `tensorflow` nor `keras` in `sys.modules`. There are zero
CRITICAL and zero HIGH advisories across 159 packages.

**A note on the verdict.** By the letter of the rubric this project scores `healthy`: no
CRITICAL/HIGH advisories, a working test runner, no high-severity configuration gaps. I have marked
it `needs-attention` anyway, because of finding #1 — a numpy deprecation that fires on the exact
artifact-loading path the planned dashboard is built on, inside a version range the project already
permits, and which the project's own warning filter does not catch. Calling that "healthy" because
it has not broken yet would be technically defensible and practically misleading. If you judge that
risk differently, the underlying findings are unchanged and the verdict is the only thing that moves.

**What to do in what order.** Finding #1 is the one worth acting on before dashboard work starts —
adding the `joblib` warning filter alone converts a future silent break into a failing test, which
is five minutes well spent. Findings #2 and #5 pair naturally as a single dev-tooling bump. Finding
#4 is trivial and directly affects how an agent behaves in this repository. And the highest-priority
item overall still sits in the stack assessment rather than here: `CLAUDE.md` and `.cursorrules`
both instruct an agent that no frontend may exist, which will fight every session of the planned
work until it is corrected.

**Next step:** address findings #1 and #4 plus the instruction-file contradiction, then proceed to
implementation planning for the dashboard change.
