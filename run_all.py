#!/usr/bin/env python3
"""One-click runner for the D-MAKER experimental suite.

Run with:
    python run_all.py

Outputs are written to ``figures/`` (PDF) and ``tables/`` (LaTeX-style text).
"""

from experiments import preliminary, ablation, comparative, extended


def main():
    print("=" * 60)
    print("D-MAKER Experimental Framework")
    print("=" * 60)
    preliminary.run()
    ablation.run()
    comparative.run()
    extended.run()
    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
