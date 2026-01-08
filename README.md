# stat-guard

**stat-guard** is a production-grade **statistical assumption validation library** for experiments such as A/B tests and controlled studies.

It acts as a **guardrail**, validating **data integrity and statistical assumptions** *before* any analysis is performed.

> **If stat-guard fails, the experiment must not be analyzed.**

---

## 🚦 Why stat-guard exists

Most statistical failures do not come from incorrect formulas.  
They come from **broken data and violated assumptions**:

- Duplicate users counted multiple times
- Users appearing in both control and treatment
- Samples too small to be meaningful
- Imbalanced or biased groups
- Metrics with zero variance
- Silent assumption violations in production pipelines

These issues often surface **after results are shipped**.

**stat-guard prevents that.**

---

## 🧠 What stat-guard does

- Validates **unit integrity** (missing IDs, duplicates, leakage)
- Checks **minimum sample size**
- Detects **group imbalance**
- Measures **covariate balance (SMD)**
- Flags **zero-variance metrics**
- Diagnoses **distribution issues** (skewness, normality)
- Separates **errors** (blocking) from **warnings** (diagnostic)
- Produces **deterministic, machine-readable reports**

Designed for:
- CI/CD pipelines
- Experiment gating systems
- Production data workflows

---

## 🚫 What stat-guard does NOT do

stat-guard is **not** a statistics engine.

It deliberately does **not**:
- ❌ Run hypothesis tests
- ❌ Modify or auto-fix data
- ❌ Apply transformations
- ❌ Guess intent or apply heuristics

This keeps behavior **explicit, transparent, and reproducible**.

---

## 🧱 Core Philosophy

- Explicit over implicit
- No automatic corrections
- Errors invalidate experiments; warnings do not
- Deterministic, reproducible behavior
- Production-first, not notebook-first
- Simple, readable, maintainable code

---

## 📦 Installation

### From GitHub (current)

stat-guard is currently distributed via GitHub:

```bash
pip install git+https://github.com/aaryansolankii/stat-guard.git
