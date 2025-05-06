from pathlib import Path

src_dir = Path(__file__).resolve().parent

PATHS = {
    'sim_data': Path(src_dir, '..', 'data').resolve(),
    'fig_path': Path(src_dir, '..', 'figures').resolve(),
    'index_path':Path(src_dir, '..', 'data', 'indices').resolve()
}

# Chech if the paths exist
PATHS['sim_data'].mkdir(parents=True, exist_ok=True)
PATHS['fig_path'].mkdir(parents=True, exist_ok=True)
PATHS['index_path'].mkdir(parents=True, exist_ok=True)