"""All data locations used by the demo figures."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path(r"J:\bPACNewData2026")
OLD_DATA_ROOT = Path(r"F:\bPACNewData2026")

DEMO_DIR = Path(r"J:\bPACNewData2026\FRINGE APP OVERVIEW\demo")
CHANA_TIF = DEMO_DIR / "ChanA_stk.tif"
CHANA_V4_DIR = Path(r"J:\bPACNewData2026\Haj Grant Example\DATA\ChanA\defringe_v4")
CHANA_V4_CLEANED = CHANA_V4_DIR / "ChanA_stk_defringed_v4.tif"
VIDEO_FPS = 300

FIG_SIZE = (13.333, 7.5)
FIG_DPI = 150
