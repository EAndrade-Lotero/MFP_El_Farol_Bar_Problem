from pathlib import Path

src_dir = Path(__file__).resolve().parent

PATHS = {
    'sim_data': Path(src_dir, '..', 'data'),
}