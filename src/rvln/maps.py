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


KNOWN_TASK_DIR_NAMES: set[str] = {m.task_dir_name for m in SUPPORTED_MAPS.values()}


def validate_task_map(task_path: str, map_info: MapInfo) -> None:
    """Raise SystemExit if the task path implies a map that differs from the simulator's.

    Task paths are structured as <map_task_dir>/<task>.json (e.g.
    greek_island/first_task.json). If the first path component is a known map
    directory and it does not match map_info.task_dir_name, the script exits
    with a helpful error telling the user which --scene to start the simulator
    with.
    """
    path = Path(task_path)
    if path.is_absolute() or path.parent == Path("."):
        return
    task_map_dir = path.parts[0]
    if task_map_dir not in KNOWN_TASK_DIR_NAMES:
        return
    if task_map_dir != map_info.task_dir_name:
        expected_scene = next(
            (m.name for m in SUPPORTED_MAPS.values() if m.task_dir_name == task_map_dir),
            task_map_dir,
        )
        raise SystemExit(
            f"Map mismatch: task '{task_path}' is for map '{expected_scene}', "
            f"but the simulator is running '{map_info.name}'.\n"
            f"Restart the simulator with:  python scripts/run_simulator.py --scene {expected_scene}"
        )
