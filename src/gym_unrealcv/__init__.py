__version__ = "2.0.3"
from gymnasium.envs.registration import register
import logging
import os
import re
from gym_unrealcv.envs.utils.misc import load_env_setting
logger = logging.getLogger(__name__)
use_docker = False

# Maps, tasks, observations, and actions used by the combinatorial registrations.
maps = [
    'Greek_Island', 'supermarket', 'Brass_Gardens', 'Brass_Palace', 'Brass_Streets',
    'EF_Gus', 'EF_Lewis_1', 'EF_Lewis_2', 'EF_Grounds', 'TemplePlaza', 'Eastern_Garden', 'Western_Garden', 'Colosseum_Desert',
    'Desert_ruins', 'SchoolGymDay', 'Venice', 'VictorianTrainStation', 'Stadium', 'IndustrialArea', 'ModularBuilding',
    'DowntownWest', 'TerrainDemo', 'InteriorDemo_NEW', 'AncientRuins', 'Grass_Hills', 'ChineseWaterTown_Ver1',
    'ContainerYard_Night', 'ContainerYard_Day', 'Old_Factory_01', 'racing_track', 'Watermills', 'WildWest',
    'SunsetMap', 'Hospital', 'Medieval_Castle', 'Real_Landscape', 'UndergroundParking', 'Demonstration_Castle',
    'Demonstration_Cave', 'PlatFormHangar', 'PlatformFactory', 'demonstration_BUNKER', 'Arctic', 'Medieval_Daytime',
    'Medieval_Nighttime', 'ModularGothic_Day', 'ModularGothic_Night',
    'UltimateFarming', 'RuralAustralia_Example_01', 'RuralAustralia_Example_02', 'RuralAustralia_Example_03',
    'LV_Soul_Cave', 'Dungeon_Demo_00', 'SwimmingPool', 'DesertMap', 'RainMap', 'SnowMap', 'ModularVictorianCity',
    'SuburbNeighborhood_Day', 'SuburbNeighborhood_Night', 'Storagehouse', 'ModularNeighborhood',
    'ModularSciFiVillage', 'ModularSciFiSeason1', 'LowPolyMedievalInterior_1', 'QA_Holding_Cells_A', 'ParkingLot', 'Demo_Roof', 'MiddleEast', 'Lighthouse',
    'Cabin_Lake', 'UniversityClassroom', 'Tokyo', 'CommandCenter', 'JapanTrainStation_Optimised', 'Hotel_Corridor', 'Museum', 'ForestGasStation',
    'KoreanPalace', 'CourtYard', 'Chinese_Landscape_Demo', 'EnglishCollege', 'OperaHouse', 'AsianTemple', 'Pyramid', 'PlanetOutDoor',
    'Map_ChemicalPlant_1', 'Hangar', 'Science_Fiction_valley_town', 'RussianWinterTownDemo01', 'LookoutTower', 'LV_Bazaar', 'OperatingRoom',
    'PostSoviet_Village', 'Old_Town', 'AsianMedivalCity', 'StonePineForest', 'TemplesOfCambodia_01_01_Exterior', 'AbandonedDistrict'
]
Tasks = ['Rendezvous', 'Rescue', 'Track', 'Navigation', 'NavigationMulti']
Observations = ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose', 'MaskDepth', 'ColorMask']
Actions = ['Discrete', 'Continuous', 'Mixed']

_registered_ids = set()


def register_env(env_id):
    """Register a single environment by ID on demand, instead of bulk-registering 93K+ combos at import."""
    if env_id in _registered_ids:
        return
    _registered_ids.add(env_id)

    # --- Robot Arm: UnrealArm-{action}{obs}-v{version} ---
    m = re.fullmatch(r'UnrealArm-(\w+?)(Pose|Color|Depth|Rgbd)-v(\d+)', env_id)
    if m:
        action, obs, version = m.group(1), m.group(2), int(m.group(3))
        register(id=env_id, entry_point='gym_unrealcv.envs:UnrealCvRobotArm_reach',
                 kwargs={'setting_file': os.path.join('robotarm', 'robotarm_reach.json'),
                         'action_type': action, 'observation_type': obs, 'docker': use_docker, 'version': version},
                 max_episode_steps=100)
        return

    # --- Tracking spline: UnrealTrack-{env}{target}{path}-{action}{obs}-v{reset} ---
    m = re.fullmatch(r'UnrealTrack-(City[12])(Malcom|Stefani)(Path[12])-(Discrete|Continuous)(Color|Depth|Rgbd)-v(\d+)', env_id)
    if m:
        env, target, path, action, obs, reset_i = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), int(m.group(6))
        reset_type = ['Static', 'Random'][reset_i]
        register(id=env_id, entry_point='gym_unrealcv.envs:UnrealCvTracking_spline',
                 kwargs={'setting_file': os.path.join('tracking', 'v0', f'{env}{target}{path}.json'),
                         'reset_type': reset_type, 'action_type': action, 'observation_type': obs,
                         'reward_type': 'distance', 'docker': use_docker},
                 max_episode_steps=3000)
        return

    # --- Multi-cam (MCRoom/Garden/UrbanTree): Unreal{env}-{action}{obs}{nav}-v{reset} ---
    mc_envs = ['MCRoom', 'Garden', 'UrbanTree']
    mc_navs = ['Random', 'Goal', 'Internal', 'None', 'RandomInterval', 'GoalInterval', 'InternalInterval', 'NoneInterval']
    mc_obs = ['Color', 'Depth', 'Rgbd', 'Gray']
    m = re.fullmatch(r'Unreal(' + '|'.join(mc_envs) + r')-(Discrete|Continuous)(' + '|'.join(mc_obs) + r')(' + '|'.join(mc_navs) + r')-v(\d+)', env_id)
    if m:
        env, action, obs, nav, reset_i = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        register(id=env_id, entry_point='gym_unrealcv.envs:UnrealCvMC',
                 kwargs={'setting_file': os.path.join('tracking', 'multicam', f'{env}.json'),
                         'reset_type': reset_i, 'action_type': action, 'observation_type': obs,
                         'reward_type': 'distance', 'docker': use_docker, 'nav': nav},
                 max_episode_steps=500)
        return

    # --- Multi-cam MCMT: UnrealMC{env}-{action}{obs}{nav}-v{reset} ---
    mcmt_envs = ['FlexibleRoom', 'Garden', 'UrbanTree']
    mcmt_navs = ['Random', 'Goal', 'Internal', 'None', 'RandomInterval', 'GoalInterval', 'InternalInterval']
    m = re.fullmatch(r'UnrealMC(' + '|'.join(mcmt_envs) + r')-(Discrete|Continuous)(' + '|'.join(mc_obs) + r')(' + '|'.join(mcmt_navs) + r')-v(\d+)', env_id)
    if m:
        env, action, obs, nav, reset_i = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        register(id=env_id, entry_point='gym_unrealcv.envs:UnrealCvMultiCam',
                 kwargs={'setting_file': os.path.join('tracking', 'mcmt', f'{env}.json'),
                         'reset_type': reset_i, 'action_type': action, 'observation_type': obs,
                         'reward_type': 'distance', 'docker': use_docker, 'nav': nav},
                 max_episode_steps=500)
        return

    # --- Task-oriented: Unreal{Task}-{map}-{action}{obs}-v{reset} ---
    actions_pattern = '|'.join(Actions)
    obs_pattern = '|'.join(Observations)
    tasks_pattern = '|'.join(Tasks)
    maps_pattern = '|'.join(re.escape(m) for m in maps)
    m = re.fullmatch(r'Unreal(' + tasks_pattern + r')-(' + maps_pattern + r')-(' + actions_pattern + r')(' + obs_pattern + r')-v(\d+)', env_id)
    if m:
        task, env, action, obs, reset_i = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        setting_file = os.path.join(task, f'{env}.json')
        max_steps = 1000 if task == 'Navigation' else 500
        register(id=env_id, entry_point=f'gym_unrealcv.envs:{task}',
                 kwargs={'env_file': setting_file, 'action_type': action, 'observation_type': obs, 'reset_type': reset_i},
                 max_episode_steps=max_steps)
        return

    # --- UnrealAgent base: UnrealAgent-{map}-{action}{obs}-v{reset} ---
    m = re.fullmatch(r'UnrealAgent-(' + maps_pattern + r')-(' + actions_pattern + r')(' + obs_pattern + r')-v(\d+)', env_id)
    if m:
        env, action, obs, reset_i = m.group(1), m.group(2), m.group(3), int(m.group(4))
        register(id=env_id, entry_point='gym_unrealcv.envs:UnrealCv_base',
                 kwargs={'setting_file': os.path.join('env_config', f'{env}.json'),
                         'action_type': action, 'observation_type': obs, 'reset_type': reset_i},
                 max_episode_steps=500)
        return

    # --- Fallback: UnrealTrack-{map}-{action}{obs}-v{reset} (task=Track for arbitrary maps) ---
    m = re.fullmatch(r'UnrealTrack-(' + maps_pattern + r')-(' + actions_pattern + r')(' + obs_pattern + r')-v(\d+)', env_id)
    if m:
        env, action, obs, reset_i = m.group(1), m.group(2), m.group(3), int(m.group(4))
        register(id=env_id, entry_point='gym_unrealcv.envs:Track',
                 kwargs={'env_file': os.path.join('Track', f'{env}.json'),
                         'action_type': action, 'observation_type': obs, 'reset_type': reset_i},
                 max_episode_steps=500)
        return

    raise ValueError(f"Unknown gym_unrealcv environment ID: {env_id}")
