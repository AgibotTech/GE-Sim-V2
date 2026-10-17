#!/usr/bin/env python3
"""Run a video-only GE-Sim checkpoint on complete WorldOracle episodes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gesim.worldoracle.inference import main
if __name__ == '__main__':
    main()
