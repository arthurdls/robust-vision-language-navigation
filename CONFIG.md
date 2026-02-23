# rvln-adls configuration

## Layout

- **config/envs/** – Optional JSON configs for each Unreal env (binary path, map, agents). Not used by the current scripts; only `config/uav_flow_envs/` is.
- **config/uav_flow_envs/** – Overlay configs for UAV-Flow-Eval (e.g. `Track/DowntownWest.json`) so the real environment uses the Linux binary under `envs/UnrealZoo-UE4/` without editing UAV-Flow. `UnrealEnv` must point at `envs/` where `UnrealZoo-UE4` lives.
- **config/defaults.yaml** – Optional default env id (e.g. `default_env_id: UnrealTrack-DowntownWest-ContinuousColor-v0`). Not required by the current scripts.
- **envs/** – Unreal env binaries. Set `UnrealEnv` to this directory (or leave unset; the sim sets it to `repo_root/envs` by default). Put the UnrealZoo-UE4 download here (e.g. `envs/UnrealZoo-UE4/Collection_v4_LinuxNoEditor/...`).
- **tasks/uav_flow_tasks/** – Task JSONs for the UAV-Flow evaluator. Populate by copying from `UAV-Flow/UAV-Flow-Eval/test_jsons/` if needed. `start_openvla_sim.py` reads from here.
- **tasks/ltl_tasks/** – Task JSONs for the LTL runner. `run_openvla_ltl.py` reads from here (e.g. `--task first_task.json` or `--run_all_tasks`).
- **results/** – Output root. UAV-Flow sim writes to `results/uav_flow_results/` (trajectory logs and plots). LTL runs write to `results/ltl_results/run_<timestamp>/` (frames, trajectory, run_info). Playback auto-detects under `results/` or you pass a directory.
- **weights/** – Model checkpoints. Put the OpenVLA checkpoint (HF format) in `weights/` or a subdir (e.g. `weights/OpenVLA-UAV/`). If you use a subdir, the server script auto-detects it when given `weights/`. If the repo uses Git LFS for `*.safetensors`, run `git lfs pull` in the checkout so the shard files are present.
- **scripts/start_openvla_server.py** – Start the OpenVLA server from repo root; uses `weights/` by default (if it contains one subdir with `config.json`, that subdir is used). Override with `--model-dir`, `--port` (default 5007), or `--gpu-id` (default 0). Loads the original UAV-Flow server code (UAV-Flow/OpenVLA-UAV/vla-scripts/openvla_act.py) with no modifications.

## Running the simulator

1. Start the OpenVLA server (from repo root, conda env `rvln-server`):  
   `python scripts/start_openvla_server.py`  
   (Uses `weights/` by default; optional: `--port 5007 --gpu-id 0`.)

2. Run the UAV-Flow sim (from repo root, conda env `rvln-sim`):  
   `python scripts/start_openvla_sim.py`  
   (Reads from `tasks/uav_flow_tasks/`, writes to `results/uav_flow_results/`. Pass `-p`, `-m`, etc. to override port or max steps; other args are forwarded to batch_run_act_all.)

3. View results:  
   `python scripts/playback_fpv.py`  
   (Auto-detects latest LTL run under `results/ltl_results/` or UAV-Flow plots under `results/`; or pass a directory, e.g. `results/ltl_results/run_2026_02_23_15_53_54/`.)

## LTL runner (run_openvla_ltl.py)

Same server as above; then run the LTL-based control loop (subgoals + goal monitoring, frames saved under `results/ltl_results/`):

- **Single task from tasks/ltl_tasks/:**  
  `python scripts/run_openvla_ltl.py --task first_task.json`
- **All tasks in tasks/ltl_tasks/:**  
  `python scripts/run_openvla_ltl.py --run_all_tasks`
- **Ad-hoc command (requires --initial-position):**  
  `python scripts/run_openvla_ltl.py -c "Go to the red building..." --initial-position 100,100,100,61`

Uses `ai_framework` for `LLM_User_Interface` and the LTL planner. To update the LTL/LLM stack: replace or update the `ai_framework` directory (and the `spot` dependency).

**Utility:** `scripts/probe_cameras.py` – list cameras and save a frame from each; outputs to `probe_results/<camera_number>/`.
