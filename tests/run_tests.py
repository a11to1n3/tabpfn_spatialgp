"""Minimal test runner (no pytest needed): python tests/run_tests.py. The test files also work with pytest."""
import importlib.util
import sys
import time
import traceback
import unittest
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent; failed = skipped = passed = 0
    for path in sorted(here.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.time()
            try:
                getattr(mod, name)(); passed += 1; print(f"PASS {path.stem}.{name} ({time.time() - t0:.1f} s)")
            except unittest.SkipTest as e:
                skipped += 1; print(f"SKIP {path.stem}.{name}: {e}")
            except Exception:
                failed += 1; print(f"FAIL {path.stem}.{name}"); traceback.print_exc()
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
