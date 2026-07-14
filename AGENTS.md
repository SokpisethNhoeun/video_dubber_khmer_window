# AGENTS.md

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

### PyQt6 Unit Testing Conventions
* **Avoid GUI Instantiation**: Do not instantiate `QMainWindow` or `QWidgets` multiple times or inside headlessly run tests, as this can crash/abort Qt.
* **Mock-Based Tests**: Prefer pure Python mock-based tests using `AppWindow.__new__(AppWindow)` and mocking methods, components, or properties instead of invoking the full `__init__` constructor.

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

# PyQt6 & Desktop Architecture Conventions

## 1. Frozen Application Support (PyInstaller)
* **Executable Resolution**: When compiled via PyInstaller, `sys.executable` points to the application binary itself.
* **Subprocess Re-run Mitigation**: Subprocesses (like Demucs or other runtimes) launched using `sys.executable -m ...` will re-run the bootloader and trigger the GUI unless argument interception (`-m`) is handled at the very top of `main.py` (running via `runpy.run_module` and exiting).

## 2. Preventing Double-Click Bugs (Synchronous API Calls)
* **Immediate Repaint**: Disable action buttons immediately, then call `QApplication.processEvents()` to force a UI repaint before executing any synchronous/blocking operations.
* **Guaranteed Recovery**: Wrap the blocking call in a `try...finally` block to guarantee restoration of the button state (re-enabling it and resetting the label).

## 3. Dialog Constructor UI Freezes
* **No Sync Network/Blocking in `__init__`**: Avoid synchronous network calls or heavy computations in dialog constructors or show events to prevent the UI from freezing on launch.
* **Asynchronous Deferral**: Perform licensing, activation, or downloader status checks asynchronously using background threads (`QThread` + worker) or timers (`QTimer.singleShot`).

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

# Improve AGENTS.md

If you discover better architecture patterns, reusable workflows, or project conventions, recommend updates instead of modifying this file automatically.
