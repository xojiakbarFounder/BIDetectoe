#!/usr/bin/env python
"""
init_db.py
----------
Run once to create the PostgreSQL schema.

    python scripts/init_db.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger
from core.database import init_db

if __name__ == "__main__":
    logger.info("Initialising database …")
    try:
        init_db()
        logger.success("Database initialised successfully.")
    except Exception as exc:
        logger.error(f"Database init failed: {exc}")
        sys.exit(1)
