from pathlib import Path

_this = Path(__file__).resolve()
MCFLY_SRC = _this.parent.parent
BIFF_SRC = _this.parent.parent.parent / 'Biff'
ROOT = MCFLY_SRC.parent
PLOTS = ROOT / 'plots'