#!/usr/bin/env python3
"""Backward-compatible entry point for the declared-option static baseline.

The former version fitted constant D1-D5 weights, which is not the requested
static GMCR definition.  This entry point now delegates to the declared
option-priority implementation.
"""

from run_static_strategy_baseline import main


if __name__ == "__main__":
    raise SystemExit(main())
