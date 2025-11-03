from __future__ import annotations

import pathlib

try:
    from .data_loader import DATA_URL, ensure_dataset
except ImportError:  # pragma: no cover - direct execution fallback
    from data_loader import DATA_URL, ensure_dataset


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parent
    csv_path = ensure_dataset(project_root)
    print(f"Dataset available at {csv_path}")
    print(f"Source URL: {DATA_URL}")


if __name__ == "__main__":
    main()
