"""
Centralized registry of supported simulation maps.

All scripts use resolve_map() to get the active MapInfo, either from
a --scene CLI flag or via interactive prompt.
"""

from dataclasses import dataclass
from pathlib import Path

_SCENES_DIR = Path(__file__).resolve().parent / "sim" / "scenes"


@dataclass(frozen=True)
class MapInfo:
    name: str
    env_id: str
    overlay_json: Path
    default_position: str
    task_dir_name: str


SUPPORTED_MAPS: dict[str, MapInfo] = {
    "DowntownWest": MapInfo(
        name="DowntownWest",
        env_id="UnrealTrack-DowntownWest-ContinuousColor-v0",
        overlay_json=_SCENES_DIR / "Track" / "DowntownWest.json",
        default_position="-600,-1270,128,61",
        task_dir_name="downtown_west",
    ),
    "Greek_Island": MapInfo(
        name="Greek_Island",
        env_id="UnrealTrack-Greek_Island-ContinuousColor-v0",
        overlay_json=_SCENES_DIR / "Track" / "Greek_Island.json",
        default_position="-2209,-4636,409,0",
        task_dir_name="greek_island",
    ),
    "SuburbNeighborhood_Day": MapInfo(
        name="SuburbNeighborhood_Day",
        env_id="UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0",
        overlay_json=_SCENES_DIR / "Track" / "SuburbNeighborhood_Day.json",
        default_position="1130,-50,143,0",
        task_dir_name="suburb_neighborhood_day",
    ),
}


def get_map(name: str) -> MapInfo:
    """Look up a map by name. Raises ValueError if not found."""
    if name in SUPPORTED_MAPS:
        return SUPPORTED_MAPS[name]
    valid = ", ".join(SUPPORTED_MAPS.keys())
    raise ValueError(f"Unknown map '{name}'. Supported maps: {valid}")


def prompt_map_selection() -> MapInfo:
    """Interactive prompt: print numbered map list, read choice from stdin."""
    maps = list(SUPPORTED_MAPS.values())
    print("\nAvailable maps:")
    for i, m in enumerate(maps, 1):
        print(f"  {i}) {m.name}")
    while True:
        try:
            choice = input("\nSelect a map [1-{}]: ".format(len(maps))).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(maps):
                selected = maps[idx]
                print(f"Selected: {selected.name}")
                return selected
        except (ValueError, EOFError):
            pass
        print(f"Invalid choice. Enter a number from 1 to {len(maps)}.")


def resolve_map(scene_arg: str | None) -> MapInfo:
    """Return MapInfo from a --scene flag value, or prompt if None."""
    if scene_arg is not None:
        return get_map(scene_arg)
    return prompt_map_selection()
