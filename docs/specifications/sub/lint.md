# Sub-spec: Lint Quality Gates

**Parent:** [pv-spec.md](../pv-spec.md) Section 5

---

## 1. Design Goals

`pv lint` is the single-command quality gate for contract codebases.
It combines validate + audit + score into one pass and adds 10
quality-gate features drawn from mature static analysis tooling
(SonarQube, Clippy, Semgrep, golangci-lint, trunk).

Design goals:

1. **SARIF-native** — machine-readable output for IDE and CI toolchain
   integration (OASIS SARIF v2.1.0)
2. **Diff-aware** — only lint changed contracts to keep CI fast
3. **Suppressible** — per-finding and per-rule suppression with audit trail
4. **Configurable** — rule severity levels tunable per-project
5. **Actionable** — auto-fix suggestions, not just diagnostics
6. **CI-integrated** — GitHub PR annotations, exit codes, trend tracking
7. **Incremental** — cache results, skip unchanged contracts
8. **Persistent** — quality trends over time for drift detection

---

## 2. SARIF Output

### Motivation

SARIF (Static Analysis Results Interchange Format) is the OASIS
standard for tool-agnostic static analysis output. GitHub Code
Scanning, VS Code SARIF Viewer, Azure DevOps, and Semgrep all
consume SARIF natively. Emitting SARIF makes `pv lint` a drop-in
for any CI/CD pipeline that already processes static analysis.

**Reference:** OASIS SARIF v2.1.0 (2020). *Static Analysis Results
Interchange Format.* docs.oasis-open.org/sarif/sarif/v2.1.0

**Reference:** Feist, J. et al. (2024). "Integrating Static Code
Analysis Toolchains." arXiv:2403.05986 — demonstrates SARIF as the
convergence format for multi-tool analysis pipelines.

### Schema Mapping

| pv lint concept | SARIF element |
|---|---|
| Contract file | `artifact` |
| Validation error | `result` with `level: "error"` |
| Audit gap | `result` with `level: "warning"` |
| Score below threshold | `result` with `level: "error"` |
| Provability violation | `result` with `level: "error"` |
| Auto-fix suggestion | `result.fixes[].artifactChanges` |
| Rule definition | `tool.driver.rules[]` |
| Rule severity | `rule.defaultConfiguration.level` |
| Suppression | `result.suppressions[]` |

### Output Format

```bash
pv lint contracts/ -f sarif > results.sarif
pv lint contracts/ -f sarif --pretty   # Human-readable SARIF
```

```json
{
  "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "pv-lint",
        "version": "0.1.0",
        "informationUri": "https://github.com/paiml/provable-contracts",
        "rules": [
          {
            "id": "PV-VAL-001",
            "shortDescription": { "text": "Missing required field" },
            "defaultConfiguration": { "level": "error" }
          },
          {
            "id": "PV-AUD-001",
            "shortDescription": { "text": "Obligation without falsification test" },
            "defaultConfiguration": { "level": "warning" }
          }
        ]
      }
    },
    "results": [
      {
        "ruleId": "PV-VAL-001",
        "level": "error",
        "message": { "text": "Contract missing proof_obligations section" },
        "locations": [{
          "physicalLocation": {
            "artifactLocation": { "uri": "contracts/example-v1.yaml" },
            "region": { "startLine": 1, "startColumn": 1 }
          }
        }]
      }
    ]
  }]
}
```

### CI Integration

```yaml
# GitHub Code Scanning
- name: Lint contracts
  run: pv lint contracts/ -f sarif > results.sarif
- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
```

---

## 3. Diff-Aware / Baseline Mode

### Motivation

Linting 165+ contracts on every CI run is wasteful. Diff-aware mode
only lints contracts changed since a base ref, cutting CI time
proportional to the change size. Baseline mode suppresses all
pre-existing findings so teams can adopt `pv lint` incrementally
without drowning in legacy warnings.

**Reference:** Nachman, L. et al. (2025). "Dealing with SonarQube
Cloud." arXiv:2508.18816 — documents baseline/new-code patterns
in industrial quality gate adoption.

### Diff-Aware

```bash
# Only lint contracts changed since main
pv lint contracts/ --diff main

# Since a specific commit
pv lint contracts/ --diff abc123

# Since last tag
pv lint contracts/ --diff v0.2.0
```

**Algorithm:**

```
1. git diff --name-only <base_ref>..HEAD -- contracts/
2. Filter to *.yaml files
3. Add transitive dependents (if A changed and B depends_on A, lint B)
4. Run full lint pipeline on filtered set only
```

### Baseline Mode

```bash
# Generate baseline from current findings
pv lint contracts/ -f sarif > .pv/baseline.sarif

# Lint with baseline — only new findings are errors
pv lint contracts/ --baseline .pv/baseline.sarif
```

**Matching algorithm:** Findings are matched by (ruleId, artifactUri,
message fingerprint). A finding present in the baseline is demoted to
`kind: "suppressed"` with `suppressionKind: "inSource"`.

---

## 4. Per-Finding Suppression

### Motivation

Not every lint finding is actionable. Per-finding suppression with an
audit trail prevents suppression from being a "make it go away" button.

**Reference:** Nachman et al. (2025). arXiv:2508.18816 — suppression
management patterns from SonarQube's "won't fix" and "false positive"
resolution workflows.

### YAML Inline Suppression

```yaml
# In the contract file:
proof_obligations:
  - id: SM-INV-001
    type: invariant
    description: "Output sums to 1.0"
    # pv-lint-suppress: PV-AUD-001
    # reason: "Covered by Lean proof SM-LEAN-001, not probar"
```

### CLI Suppression

```bash
# Suppress specific finding IDs
pv lint contracts/ --suppress SM-INV-001,KANI-SM-002

# Suppress by rule
pv lint contracts/ --suppress-rule PV-AUD-001

# Suppress file
pv lint contracts/ --suppress-file contracts/arch-constraints-v1.yaml
```

### Suppression Registry

Suppressions are tracked in `.pv/suppressions.yaml`:

```yaml
suppressions:
  - finding: SM-INV-001
    rule: PV-AUD-001
    reason: "Covered by Lean proof, tracked in PMAT-072"
    author: "noah"
    date: "2026-03-07"
    expires: "2026-06-07"  # Optional TTL
```

Expired suppressions re-surface automatically.

---

## 5. Configurable Rule Severity

### Motivation

Different projects have different maturity levels. A research
prototype may treat missing Kani harnesses as warnings; a production
stack treats them as errors.

### Severity Levels

| Level | Behavior | Default rules |
|---|---|---|
| `error` | Fails `--exit-code` | Validation errors, provability violations |
| `warning` | Reported, does not fail | Audit gaps, low scores |
| `info` | Informational only | Style suggestions, paper ref gaps |
| `off` | Disabled entirely | User choice |

### Configuration

In `.pv.toml` (see Section 11):

```toml
[lint.rules]
PV-VAL-001 = "error"     # Missing required field
PV-AUD-001 = "warning"   # Obligation without test
PV-AUD-002 = "info"      # Missing paper reference
PV-SCR-001 = "error"     # Score below threshold
PV-PRV-001 = "error"     # Provability invariant violation
```

CLI override:

```bash
# Only show errors
pv lint contracts/ --severity error

# Promote warnings to errors (strict mode)
pv lint contracts/ --strict

# Demote a rule for this run
pv lint contracts/ --rule PV-AUD-001=info
```

---

## 6. Auto-Fix Suggestions

### Motivation

The best lint finding is one that fixes itself. Where a fix is
deterministic, `pv lint` emits SARIF `fix` objects with concrete
`artifactChanges`.

**Reference:** Yang, J. et al. (2025). "CodeCureAgent: Automatic
Classification and Repair of Static Analysis Warnings."
arXiv:2509.11787 — automated classification and repair of static
analysis findings.

**Reference:** Shestov, A. et al. (2025). "Augmenting LLMs with
Static Code Analysis for Automated Code Quality Improvements."
arXiv:2506.10330 — combining static analysis with LLM-powered
auto-fix for quality improvement.

### Fixable Rules

| Rule | Fix |
|---|---|
| PV-VAL-002: Missing `metadata.version` | Insert `version: "1.0.0"` |
| PV-VAL-003: Missing `metadata.created` | Insert today's date |
| PV-AUD-003: Obligation ID not in tests | Generate test stub ID |
| PV-SCR-002: Missing binding entry | Add `not_implemented` stub |

### Usage

```bash
# Show suggestions (dry run)
pv lint contracts/ --suggest

# Apply fixes (modifies files)
pv lint contracts/ --fix

# Apply fixes and show diff
pv lint contracts/ --fix --diff
```

### SARIF Fix Object

```json
{
  "ruleId": "PV-VAL-002",
  "fixes": [{
    "description": { "text": "Add version field" },
    "artifactChanges": [{
      "artifactLocation": { "uri": "contracts/example-v1.yaml" },
      "replacements": [{
        "deletedRegion": { "startLine": 2, "startColumn": 1 },
        "insertedContent": { "text": "  version: \"1.0.0\"\n" }
      }]
    }]
  }]
}
```

---

## 7. GitHub PR Annotations

### Motivation

Developers see lint results where they work — in the PR diff view.
SARIF upload to GitHub Code Scanning provides this, but native
`--format github` emits GitHub Actions workflow commands for
zero-config annotation.

### Usage

```bash
# GitHub Actions workflow commands (::error, ::warning)
pv lint contracts/ -f github

# Output:
# ::error file=contracts/example-v1.yaml,line=1::PV-VAL-001: Missing proof_obligations
# ::warning file=contracts/softmax-kernel-v1.yaml,line=42::PV-AUD-001: SM-INV-003 has no test
```

### GitHub Actions Integration

```yaml
- name: Lint contracts
  run: pv lint contracts/ -f github --diff ${{ github.event.pull_request.base.sha }}
```

Combined with SARIF upload for persistent tracking:

```yaml
- name: Lint contracts (annotations + SARIF)
  run: |
    pv lint contracts/ -f github --diff ${{ github.base_ref }}
    pv lint contracts/ -f sarif > results.sarif
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
```

---

## 8. Watch Mode

### Motivation

During contract authoring, instant feedback on save accelerates the
write-validate-fix loop.

### Usage

```bash
# Watch contracts/ for changes, re-lint on save
pv lint contracts/ --watch

# Watch with filters
pv lint contracts/ --watch --severity error

# Watch specific contract
pv lint contracts/softmax-kernel-v1.yaml --watch
```

### Behavior

```
1. Perform initial full lint
2. Watch contracts/ with notify/inotify
3. On file change:
   a. Debounce 200ms
   b. Re-lint only changed file + transitive dependents
   c. Clear terminal, show results
4. On Ctrl+C: exit with last lint exit code
```

---

## 9. Trend / History Tracking

### Motivation

Quality gates answer "is it good enough now?" but not "is it getting
better?" Trend tracking answers the second question by recording
lint results over time and detecting quality drift.

**Reference:** Molnar, A. & Motogna, S. (2024). "Versioned Analysis
of Software Quality Indicators and Self-Admitted Technical Debt."
arXiv:2407.15967 — longitudinal quality metrics and SATD evolution.

### Storage

```
.pv/trend/
+-- 2026-03-07T14:30:00.json    # Timestamped snapshots
+-- 2026-03-06T09:15:00.json
+-- ...
```

Each snapshot:

```json
{
  "timestamp": "2026-03-07T14:30:00Z",
  "commit": "abc123",
  "total_contracts": 165,
  "errors": 0,
  "warnings": 12,
  "mean_score": 0.78,
  "grade_distribution": { "A": 45, "B": 60, "C": 40, "D": 15, "F": 5 },
  "unproven_obligations": 23,
  "binding_gaps": 6,
  "findings_by_rule": { "PV-AUD-001": 8, "PV-AUD-002": 4 }
}
```

### Usage

```bash
# Record current state
pv lint contracts/ --trend

# Show trend (last 30 snapshots)
pv lint contracts/ --trend --show
# Output:
# Date        Score  Errors  Warnings  Grade
# 2026-03-07  0.78   0       12        B
# 2026-03-06  0.76   1       14        B
# 2026-03-01  0.71   3       18        C
# Trend: +0.07 score over 7 days (improving)

# JSON trend for CI dashboards
pv lint contracts/ --trend --show -f json
```

### Drift Detection

If mean score drops >5% from the 7-day rolling average, `--trend`
emits a `PV-TRD-001` warning. This catches gradual quality erosion
that individual per-contract gates miss.

---

## 10. Caching / Incremental Analysis

### Motivation

Full lint of 165 contracts takes ~2s. As the registry grows toward
1000+, incremental analysis via content-addressable caching keeps
lint O(changed) rather than O(total).

**Reference:** Singh, G. et al. (2022). "Interactive Abstract
Interpretation." arXiv:2209.10445 — incremental abstract
interpretation with fixpoint caching.

### Cache Design

```
.pv/cache/
+-- lint/
    +-- <blake3-hash>.json    # Per-contract lint result
```

**Cache key:** BLAKE3 hash of (contract YAML content + binding.yaml
content + rule config). Cache is automatically invalidated when any
input changes.

### Usage

```bash
# Normal lint (uses cache)
pv lint contracts/

# Force full re-lint (bypass cache)
pv lint contracts/ --no-cache

# Show cache stats
pv lint contracts/ --cache-stats
# Output: 165 contracts, 162 cached, 3 re-linted (1.8% work)
```

### Invalidation Rules

| Change | Invalidation |
|---|---|
| Contract YAML modified | That contract |
| binding.yaml modified | All contracts with bindings |
| `.pv.toml` rules changed | All contracts |
| `--no-cache` flag | All contracts |
| Transitive dependency changed | Dependent contracts |

---

## 11. Configuration File (`.pv.toml`)

### Motivation

CLI flags are ephemeral. A checked-in config file makes lint
behavior reproducible across CI, local dev, and team members.

### Location

`pv lint` searches for `.pv.toml` in:
1. Current directory
2. Repository root (git rev-parse --show-toplevel)
3. `$HOME/.config/pv/config.toml` (user default)

CLI flags override config file values.

### Schema

```toml
[lint]
min_score = 0.60
severity = "warning"          # Minimum severity to report
strict = false                # Promote warnings to errors
contracts_dir = "contracts/"
binding = "contracts/aprender/binding.yaml"

[lint.rules]
PV-VAL-001 = "error"
PV-AUD-001 = "warning"
PV-AUD-002 = "info"
PV-SCR-001 = "error"
PV-PRV-001 = "error"

[lint.suppress]
findings = ["SM-INV-001"]
rules = ["PV-AUD-002"]
files = ["contracts/arch-constraints-v1.yaml"]

[lint.diff]
base_ref = "main"             # Default base for --diff

[lint.trend]
enabled = true
retention_days = 90
drift_threshold = 0.05

[lint.cache]
enabled = true
dir = ".pv/cache"

[output]
format = "text"               # text, json, sarif, markdown, github
color = "auto"                # auto, always, never
```

---

## 12. Rule Catalog

All `pv lint` rules follow the pattern `PV-<CATEGORY>-NNN`.

### Validation Rules (PV-VAL)

| Rule | Severity | Description |
|---|---|---|
| PV-VAL-001 | error | Missing required field |
| PV-VAL-002 | error | Missing `metadata.version` |
| PV-VAL-003 | warning | Missing `metadata.created` |
| PV-VAL-004 | error | Invalid cross-reference (obligation -> test) |
| PV-VAL-005 | error | Duplicate obligation/test ID |
| PV-VAL-006 | warning | Unreachable test (no obligation references it) |

### Audit Rules (PV-AUD)

| Rule | Severity | Description |
|---|---|---|
| PV-AUD-001 | warning | Obligation without falsification test |
| PV-AUD-002 | info | Missing paper reference |
| PV-AUD-003 | warning | Obligation ID not referenced by any test |
| PV-AUD-004 | warning | Equation without domain specification |
| PV-AUD-005 | warning | Missing tolerance for numerical obligation |

### Score Rules (PV-SCR)

| Rule | Severity | Description |
|---|---|---|
| PV-SCR-001 | error | Composite score below `--min-score` |
| PV-SCR-002 | warning | Missing binding entry for equation |
| PV-SCR-003 | info | Kani coverage below 50% |
| PV-SCR-004 | info | Lean coverage at 0% |

### Provability Rules (PV-PRV)

| Rule | Severity | Description |
|---|---|---|
| PV-PRV-001 | error | Kernel contract without Kani harnesses |
| PV-PRV-002 | error | Kernel contract without falsification tests |
| PV-PRV-003 | warning | Kani harness count < obligation count |
| PV-PRV-004 | info | No Lean theorems (L5 not attempted) |

### Trend Rules (PV-TRD)

| Rule | Severity | Description |
|---|---|---|
| PV-TRD-001 | warning | Mean score dropped >5% from 7-day rolling avg |
| PV-TRD-002 | info | Error count increased from previous snapshot |

---

## 13. Implementation Plan

### Phase 1: SARIF + Config [Priority: Highest]

- SARIF v2.1.0 output format (`-f sarif`)
- `.pv.toml` config file parsing
- Rule catalog with configurable severity
- GitHub Actions workflow command format (`-f github`)

### Phase 2: Diff-Aware + Caching

- `--diff <base_ref>` with transitive dependent expansion
- BLAKE3 content-addressable lint cache
- `--no-cache` and `--cache-stats`

### Phase 3: Suppression + Baseline

- `--baseline <sarif>` mode
- `--suppress` / `--suppress-rule` / `--suppress-file`
- `.pv/suppressions.yaml` with expiry
- YAML inline `# pv-lint-suppress` comments

### Phase 4: Auto-Fix + Suggestions

- `--suggest` (dry run)
- `--fix` (apply changes)
- SARIF `fix` objects for deterministic repairs

### Phase 5: Trend + Watch

- `--trend` snapshot recording
- `--trend --show` historical display
- `--watch` with inotify/notify debounced re-lint
- PV-TRD drift detection rules

---

## 14. References

### Standards

1. OASIS (2020). *Static Analysis Results Interchange Format (SARIF)
   Version 2.1.0.* docs.oasis-open.org/sarif/sarif/v2.1.0

### Toolchain Integration

2. Feist, J. et al. (2024). "Integrating Static Code Analysis
   Toolchains." arXiv:2403.05986

3. Nachman, L. et al. (2025). "Dealing with SonarQube Cloud:
   Investigating the Integration of a Cloud-Based Static Analysis
   Service." arXiv:2508.18816

### Auto-Fix and Repair

4. Yang, J. et al. (2025). "CodeCureAgent: Automatic Classification
   and Repair of Static Analysis Warnings." arXiv:2509.11787

5. Shestov, A. et al. (2025). "Augmenting LLMs with Static Code
   Analysis for Automated Code Quality Improvements."
   arXiv:2506.10330

### Quality Metrics

6. Molnar, A. & Motogna, S. (2024). "Versioned Analysis of Software
   Quality Indicators and Self-Admitted Technical Debt."
   arXiv:2407.15967

### Incremental Analysis

7. Singh, G. et al. (2022). "Interactive Abstract Interpretation:
   Reanalyzing Whole Programs for Cheap." arXiv:2209.10445

### Formal Contracts

8. Li, Y. et al. (2025). "Do Large Language Models Respect
   Contracts?" arXiv:2510.12047

9. Bruni, R. et al. (2026). "Agent Behavioral Contracts."
   arXiv:2602.22302
