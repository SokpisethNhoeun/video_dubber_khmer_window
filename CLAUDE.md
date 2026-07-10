# CLAUDE.md

## Project Goal

Maintain a high-quality, production-ready PyQt6 desktop application for Khmer video dubbing.

Priorities (highest to lowest):

1. Correctness
2. Stability
3. Maintainability
4. Performance
5. UX/UI
6. Clean architecture

---

# Development Workflow

For every feature, bug fix, refactor, or architectural change:

## 1. Analysis

* Read the relevant code.
* Understand the existing implementation.
* Reuse existing components whenever possible.
* Use **python-desktop-architect** for architecture analysis.
* Ask clarification questions.
* Wait for approval.

Never implement before approval.

---

## 2. Planning

After clarification:

Produce a concise implementation plan including:

* affected files
* risks
* testing strategy

Wait for approval.

---

## 3. Implementation

Follow existing architecture.

Keep changes:

* minimal
* focused
* backward compatible

Avoid unnecessary refactoring.

Explain important design decisions.

---

## 4. Testing

After significant changes run:

```bash
python -m pytest tests/
```

Report:

* passed
* failed
* skipped
* duration

If tests fail:

* identify root cause
* explain whether failures are new
* recommend fixes

---

## 5. Review

If tests pass:

Use **code-reviewer**.

Review:

* bugs
* architecture
* maintainability
* security
* performance
* edge cases
* error handling

---

## 6. QA

After code review:

Use **pyqt6-qa-ux-reviewer**.

Review:

* functionality
* UX
* UI
* threading
* responsiveness
* resource cleanup
* PyQt best practices

Report findings by severity.

---

# Project Rules

Always preserve:

* pipeline checkpoints
* session resume
* cache integrity
* cancellation support
* existing architecture

For GPU work:

* load one heavy model at a time
* free GPU memory after use
* avoid unnecessary model reloads

Never introduce breaking changes without approval.

---

# Completion Checklist

Before finishing:

* Architecture reviewed
* Questions answered
* Plan approved
* Implementation complete
* Code reviewed
* QA completed

---

# Improve CLAUDE.md

If you discover better architecture patterns, reusable workflows, or project conventions, recommend updates instead of modifying this file automatically.
