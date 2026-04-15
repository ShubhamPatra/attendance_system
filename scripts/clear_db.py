"""Clear all data from the MongoDB database."""

from __future__ import annotations

import sys

import core.database as database


def main() -> int:
    try:
        db = database.get_db()
        
        # Get all collection names
        collections = db.list_collection_names()
        
        if not collections:
            print("Database is already empty.")
            return 0
        
        print(f"Found {len(collections)} collection(s) to clear: {', '.join(collections)}")
        
        # Drop all collections
        for collection_name in collections:
            db.drop_collection(collection_name)
            print(f"  ✓ Dropped collection: {collection_name}")
        
        print("\nDatabase cleared successfully.")
        return 0
    
    except Exception as exc:
        print(f"Error clearing database: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
