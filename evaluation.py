import json
import numpy as np
from os.path import exists


save_dir = ""


def load_data(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def merge_files(file_names, out_file_name):
    logprobs = {}
    for file_name in file_names:
        logprobs = {**logprobs, **load_data(file_name)}    
    json.dump(logprobs, open(out_file_name, 'w'), indent=1)
    return


def evaluate(file_name):
    b = int(file_name.split("_")[-2][-1])
    if b:
        ret = {"master": [0, 0], "ensemble": [0, 0]}
        data = load_data(file_name)
        for comp_dicts in data.values():
            master_logprobs = []
            ensemble_logprobs = []
            for comp_dict in comp_dicts:
                master_comp_logprobs = [0,0,0,0,0,0,0,0]
                ensemble_comp_logprobs = [0,0,0,0,0,0,0,0]
                for branch_name, branch_dict in comp_dict.items():
                    for i in range(8):
                        token_logprobs = branch_dict[i]["token_logprobs"]
                        if branch_name == "master":
                            master_comp_logprobs[i] += (sum(token_logprobs)/len(token_logprobs))
                        else:
                            ensemble_comp_logprobs[i] += (sum(token_logprobs)/len(token_logprobs))
                master_logprobs.append(master_comp_logprobs)
                ensemble_logprobs.append(ensemble_comp_logprobs)
            master_logprobs = np.array(master_logprobs)
            master_logprobs = np.sum(master_logprobs, axis=0)
            if master_logprobs[0] == max(master_logprobs):
                ret["master"][0] += 1
            ret["master"][1] += 1
            ensemble_logprobs = np.array(ensemble_logprobs)
            ensemble_logprobs = np.sum(ensemble_logprobs, axis=0)
            if ensemble_logprobs[0] == max(ensemble_logprobs):
                ret["ensemble"][0] += 1
            ret["ensemble"][1] += 1
        return ret
    else:
        ret = [0,0]
        data = load_data(file_name)
        for prob_dict in data.values():
            logprobs = []
            for i in range(8):
                token_logprobs = prob_dict[i]["token_logprobs"]
                logprobs.append(sum(token_logprobs)/len(token_logprobs))
            if logprobs[0] == max(logprobs):
                ret[0] += 1
            ret[1] += 1
        return ret