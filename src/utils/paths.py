# As the data and code is moved frequently between machines, we give the paths (to the data, config file, etc...) in one of the following ways_
#  - either as an absolute path, i.e. /eos/home-g/gkrzmanc/jetclustering/code/config_files/config_jets.yaml
#  - or as a path relative to either the SVJ_CODE_ROOT, SVJ_DATA_ROOT, SVJ_PREPROCESSED_DATA_ROOT, or RESULTS_ROOT directories: config_files/config_jets.yaml
# these env_vars are set in env.sh and this file is not copied between machines, i.e. lxplus and tier3.

import os

def get_path(path, type="code"):
    assert type in ["code", "data", "preprocessed_data", "results"]
    path = path.strip()
    if path.startswith("/"):
        return path
    if type == "code":
        return os.path.join(os.environ["SVJ_CODE_ROOT"], path)
    if type == "data":
        return os.path.join(os.environ["SVJ_DATA_ROOT"], path)
    if type == "preprocessed_data":
        return os.path.join(os.environ["SVJ_PREPROCESSED_DATA_ROOT"], path)
    if type == "results":
        return os.path.join(os.environ["SVJ_RESULTS_ROOT"], path)

