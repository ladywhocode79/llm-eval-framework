#!/usr/bin/env python3
"""
Pre-flight setup checker for the LLM Eval Framework.

Run this before your first test run to verify everything is configured correctly.

Usage:
    python scripts/check_setup.py
"""

import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"

errors = []


def check(label: str, fn):
    try:
        result = fn()
        print(f"{PASS} {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"{FAIL} {label}\n         {e}")
        errors.append(label)
        return False


# ── 1. Python version ─────────────────────────────────────────────────────────
print("\n=== LLM Eval Framework — Setup Check ===\n")

version = sys.version_info
if version >= (3, 10):
    print(f"{PASS} Python version — {sys.version.split()[0]}")
else:
    print(f"{WARN} Python version — {sys.version.split()[0]} (3.10+ recommended)")

# ── 2. Required packages ──────────────────────────────────────────────────────
print()

def _import(pkg): __import__(pkg)

check("anthropic package",   lambda: _import("anthropic"))
check("deepeval package",    lambda: _import("deepeval"))
check("ollama package",      lambda: _import("ollama"))
check("pytest package",      lambda: _import("pytest"))
check("pytest-html package", lambda: _import("pytest_html"))
check("python-dotenv",       lambda: _import("dotenv"))

# ── 3. Environment variables ──────────────────────────────────────────────────
print()

from dotenv import load_dotenv
load_dotenv()

anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
if anthropic_key and anthropic_key != "your-api-key-here":
    print(f"{PASS} ANTHROPIC_API_KEY — set ({anthropic_key[:8]}...)")
else:
    print(f"{FAIL} ANTHROPIC_API_KEY — not set or still placeholder")
    print(f"       Fix: copy .env.example to .env and add your key")
    errors.append("ANTHROPIC_API_KEY")

backend = os.environ.get("JUDGE_BACKEND", "ollama")
model   = os.environ.get("OLLAMA_MODEL", "llama3.2")
print(f"{INFO} JUDGE_BACKEND  — {backend}")
print(f"{INFO} OLLAMA_MODEL   — {model}")

# ── 4. Ollama server + model (only when using ollama backend) ─────────────────
if backend == "ollama":
    print()

    try:
        import ollama

        # Server reachable?
        try:
            pulled = ollama.list().models
            pulled_names = [m.model for m in pulled]
            print(f"{PASS} Ollama server  — running")
        except Exception as e:
            print(f"{FAIL} Ollama server  — not reachable")
            print(f"       Error: {e}")
            print(f"       Fix:   run 'ollama serve' in a separate terminal")
            errors.append("Ollama server")
            pulled_names = []

        # Model pulled?
        if pulled_names is not None:
            model_base = model.split(":")[0]
            found = any(m.split(":")[0] == model_base for m in pulled_names)
            if found:
                print(f"{PASS} Ollama model   — '{model}' is available")
            else:
                print(f"{FAIL} Ollama model   — '{model}' not found")
                print(f"       Available: {pulled_names or '(none pulled yet)'}")
                print(f"       Fix:   ollama pull {model}")
                errors.append(f"Ollama model '{model}'")

    except ImportError:
        pass  # already caught above in package checks

# ── 5. Dataset file ───────────────────────────────────────────────────────────
print()

dataset_path = os.path.join(os.path.dirname(__file__), "..", "evals", "datasets", "qa_test_cases.json")
check("Test dataset file", lambda: (
    "found" if os.path.exists(dataset_path) else (_ for _ in ()).throw(FileNotFoundError(dataset_path))
))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if not errors:
    print("=== All checks passed. You're ready to run tests! ===")
    print()
    print("    pytest -m eval -v")
    print()
else:
    print(f"=== {len(errors)} issue(s) found. Fix the above before running tests. ===")
    print()
    print("    Failed checks:")
    for e in errors:
        print(f"      - {e}")
    print()
    sys.exit(1)
