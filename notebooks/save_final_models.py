# This script saves the final model weights in the "models/" directory, ready for upload to HF

final_models = [
    # dataset, loss, run name, step
    # GP_IRC_SN
    ["900_03", "GP_IRC_SN", "Delphes_Aug_IRCSplit_50k_SN_from3kFT_2025_05_16_14_07_29_474", 21060],
    ["QCD", "GP_IRC_SN", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_24_23_00_54_948", 24000],
    ["700_07+900_03+QCD", "GP_IRC_SN", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_24_23_00_56_910", 24000],
    ["700_07+900_03", "GP_IRC_SN", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_24_23_01_01_212", 24000],
    ["700_07", "GP_IRC_SN", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_24_23_01_07_703", 24000],

    # GP_IRC_S
    ["700_07+900_03", "GP_IRC_S", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_20_15_29_30_29", 24000],
    ["700_07+900_03+QCD", "GP_IRC_S", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_20_15_29_28_959", 24000],
    ["700_07", "GP_IRC_S", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_20_15_11_35_476", 24000],
    ["QCD", "GP_IRC_S", "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_20_15_11_20_735", 24000],
    ["900_03", "GP_IRC_S", "Delphes_Aug_IRCSplit_50k_from10k_2025_05_11_14_08_49_675", 9960],

    # GP
    ["900_03", "GP", "LGATr_Aug_50k_2025_05_09_15_25_32_34", 24000],
    ["700_07", "GP", "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_19_21_38_20_376", 24000],
    ["700_07+900_03", "GP", "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_20_13_13_00_503", 24000],
    ["700_07+900_03+QCD", "GP", "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_20_13_12_54_359", 24000],
    ["QCD", "GP", "GP_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_19_21_29_06_946", 24000],

    # Base training
    ["900_03", "base", "LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 50000],
    ["700_07", "base", "LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_16_19_44_46_795", 50000],
    ["QCD", "base", "LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_16_19_46_57_48", 50000],
    ["700_07+900_03", "base", "LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_16_21_04_26_991", 50000],
    ["700_07+900_03+QCD", "base", "LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_16_21_04_26_937", 50000]
]



import os
import shutil
def get_run_step_direct(run_name, step):
    # get the step of the run directly
    p = os.path.join("/pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/results/train", run_name)
    lst = os.listdir(p)
    lst = [x for x in lst if x.endswith(".ckpt")] # files are of format step_x_epoch_y.ckpt
    steps = [int(x.split("_")[1]) for x in lst]
    if step not in steps:
        print("Available steps:", steps)
        raise Exception("Step not found in run")
    full_path = os.path.join(p, [x for x in lst if int(x.split("_")[1]) == step][0])
    return full_path

if not os.path.exists("models"):
    os.makedirs("models")

for model in final_models:
    print(model)
    dataset, loss, run_name, step = model
    if not os.path.exists(os.path.join("models", loss)):
        os.makedirs("models/" + loss)
    p = get_run_step_direct(run_name, step)
    # copy  p to models/loss/dataset.ckpt
    dst_path = f"models/{loss}/{dataset}.ckpt"
    shutil.copyfileobj(open(p, "rb"), open(dst_path, "wb"))
    print("Copied", p, "to", dst_path)

