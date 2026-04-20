"""
⚠️ DEPRECATED - File moved to scripts/dev-only/

This file has been moved to maintain a clean separation between development-only
scripts and production utilities.

The seed_demo_data.py script is now located at:
    scripts/dev-only/seed_demo_data.py

To use the demo data seeding script:
    python scripts/dev_only/seed_demo_data.py
    
    OR from project root:
    
    python -m scripts.dev_only.seed_demo_data

For more information, see:
    - docs/BUILD_GUIDE.md (Step 1: Seed Demo Data)
    - docs/APPENDIX.md (Administrative Tasks)
    - scripts/dev-only/README.md
"""

import sys

def main() -> int:
    print("❌ ERROR: seed_demo_data.py has been moved!")
    print()
    print("New location: scripts/dev_only/seed_demo_data.py")
    print()
    print("To seed demo data, run:")
    print("    python scripts/dev_only/seed_demo_data.py")
    print()
    print("For more information, see:")
    print("    - docs/BUILD_GUIDE.md")
    print("    - scripts/dev-only/README.md")
    print()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())

