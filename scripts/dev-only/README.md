# Development-Only Scripts

⚠️ **WARNING: DO NOT RUN IN PRODUCTION**

This folder contains development and testing scripts that are **NOT intended for production use**.

## Contents

- `seed_demo_data.py` - Creates test/demo students and sample attendance data for local development and testing

## Usage

These scripts are only for local development environments to generate test data. 

### Example: Seed Demo Data

```bash
# Before running, ensure MongoDB is running and configured
python -m scripts.dev_only.seed_demo_data
```

Or directly:

```bash
python scripts/dev_only/seed_demo_data.py
```

## Important Notes

1. **Never run these scripts in production** - They create fake/test data only
2. **Data created is for testing only** - Do not rely on demo students for real attendance tracking
3. **Hardcoded credentials** - The seed script includes test accounts with default credentials
4. **Consider cleanup** - After testing, use `scripts/clear_db.py` to reset the database

## See Also

- Parent directory: `scripts/` - Contains production utilities
- `scripts/clear_db.py` - Clears all data from MongoDB (use with caution!)
- `docs/INSTALLATION.md` - Local development setup guide
