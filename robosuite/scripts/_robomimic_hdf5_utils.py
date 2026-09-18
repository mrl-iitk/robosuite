"""
Shared helper for the collect_scripted_*_demos.py scripts.

gather_demonstrations_as_hdf5 (in collect_human_demonstrations.py) only writes the raw
robosuite attribute `env_info` on the `data` group. robomimic's own tooling
(get_dataset_info.py, dataset_states_to_obs.py, playback_dataset.py, ...) instead reads
an `env_args` attribute (plus `total`/`num_samples`), which is normally added by a
separate manual pass through robomimic's scripts/conversion/convert_robosuite.py.
`finalize_for_robomimic` replicates that conversion in-place so the hdf5 these scripts
write is directly usable by robomimic without that extra step.
"""

import json

import h5py

# robomimic.envs.env_base.EnvType.ROBOSUITE_TYPE -- duplicated here to avoid a hard
# dependency on robomimic being installed just to collect a dataset.
ROBOMIMIC_ROBOSUITE_ENV_TYPE = 1


def finalize_for_robomimic(hdf5_path):
    """Add the `env_args`/`total`/`num_samples` metadata robomimic's tooling expects,
    mirroring robomimic/scripts/conversion/convert_robosuite.py."""
    with h5py.File(hdf5_path, "a") as f:
        data_grp = f["data"]

        env_info = json.loads(data_grp.attrs["env_info"])
        env_meta = dict(
            type=ROBOMIMIC_ROBOSUITE_ENV_TYPE,
            env_name=data_grp.attrs["env"],
            env_version=data_grp.attrs["repository_version"],
            env_kwargs=env_info,
        )
        data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)

        total_samples = 0
        for ep in data_grp:
            n_samples = data_grp[ep]["actions"].shape[0]
            data_grp[ep].attrs["num_samples"] = n_samples
            total_samples += n_samples
        data_grp.attrs["total"] = total_samples
