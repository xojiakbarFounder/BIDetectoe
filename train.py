#!/usr/bin/env python
"""Convenience wrapper: python train.py --epochs 50 --batch 16 --imgsz 640."""

from training.train import main


if __name__ == "__main__":
    main()

