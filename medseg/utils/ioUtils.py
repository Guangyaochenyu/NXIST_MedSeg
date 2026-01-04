import math
from pathlib import Path

__all__ = ['get_file_size']

get_file_size = lambda file_path, decimal_places=2, binary=True: (lambda size_bytes: (lambda units, base: f"{size_bytes / (base ** (idx := min(math.floor(math.log(size_bytes, base)) if size_bytes else 0, len(units)-1))):.{decimal_places}f} {units[idx]}")(['B', 'KB', 'MB', 'GB', 'TB'], 1024 if binary else 1000))(Path(file_path).stat().st_size)
