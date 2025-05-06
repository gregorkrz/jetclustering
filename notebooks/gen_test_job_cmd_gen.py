# This script is used as a tool to make it easier to generate test jobs from clustering training
"""
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"

python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"


python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
"""

#runs = ["LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59", "LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58", "LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59", "LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59"]

#runs1 = ["LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59", "LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58"]# Trained on 03_900, on scouting PFCands
#runs2 = ["LGATr_training_NoPIDGL_10_16_64_0.8_2025_03_17_20_05_04", "LGATr_training_NoPIDGL_10_16_64_2.0_2025_03_17_20_05_04"] # Trained on 03_900 dataset (rinv=0.3, m=900), Gen level
#runs_alldata = ["LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59", ]
# Transformer
runs_tr = ["Transformer_training_NoPID_10_16_64_0.8_2025_03_03_15_55_50", "Transformer_training_NoPID_10_16_64_2.0_2025_03_03_17_00_38"]
#runs = runs1+runs2

# Test run with no filtering
#   "LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59": "900_03",
#         "LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58": "900_03",

#runs = runs_tr
runs = ["LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59", "LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58"]
# TODO: add transformer runs into eval

#runs = runs_tr

#runs = ["LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_2025_03_27_12_46_12_740"] # test the aug. finetuning
#runs = ["LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_2025_03_27_17_09_14_641"] # R=2.0 version of the above run


runs = ["LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_36_81", "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_37_44"]

runs = ["LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_25_820", "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_26_400"]
# Parton-level training - loss seems very high, maybe something is wrong with the ds?

#runs = ["LGATr_training_NoPIDPL_10_16_64_2.0_2025_03_21_14_51_15_195"]
runs = ["LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_1e-2_2025_03_29_14_58_38_650", "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_1e-2_2025_03_29_14_58_36_446"]
runs = ["LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_2025_03_27_12_46_12_740"]

# fixed 500 particles per event and pt=1e-2
runs = ["LGATr_pt_1e-2_500part_2025_04_01_10_33_28_457"]
runs = ["LGATr_pt_1e-2_500part_2025_04_01_12_18_23_661"]

# FT on parton-level + aug clustering
runs = ["LGATr_pt_1e-2_500part_2025_04_01_16_49_08_406"]

runs = ["LGATr_pt_1e-2_500part_2025_04_01_21_14_07_350"] + ["LGATr_pt_1e-2_500part_2025_04_01_16_49_08_406"]


#  Again eval on the 'old' run that was trained on 500 ghosts with pt=0.01
runs = ["LGATr_pt_1e-2_500part_2025_04_01_10_33_28_457"]

runs = ["LGATr_pt_1e-2_500part_NoQMin_2025_04_03_23_15_17_745", "LGATr_pt_1e-2_500part_NoQMin_2025_04_03_23_15_35_810"]
runs = ["LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_37_44", "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_36_81"]


# Try to reproduce the results without qmin
runs = ["LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_2025_04_04_12_57_47_788", "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_2025_04_04_12_57_51_536"]

runs = ["LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_CW0_2025_04_04_15_30_16_839", "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_CW0_2025_04_04_15_30_20_113"]

runs = ["debug_IRC_loss_weighted100_plus_ghosts_2025_04_08_22_40_33_972"] # Short irc loss training, high coord loss - not super useful I think

runs = ["debug_IRC_loss_weighted100_plus_ghosts_2025_04_09_13_48_55_569",# Longer irc loss training, coord loss converges to a lower number - probably more useful - latest step 9960
        "LGATr_500part_NOQMin_2025_04_09_21_53_37_210"] # Reproduce the results with 500 ghosts, no qmin
#runs = ["debug_IRC_loss_weighted100_plus_ghosts_2025_04_09_13_48_55_569"] #just the 2nd one
#runs = ["debug_IRC_loss_weighted100_plus_ghosts_Qmin05_2025_04_09_14_45_51_381"] # qmin=0.5, otherwise same as above
#runs = ["debug_IRC_loss_weighted100_plus_ghosts_Qmin05_CoordLossWeight1_2025_04_09_15_29_29_203"]
#runs = ["IRC_loss_Split_and_Noise_alternate_NoAug_2025_04_11_16_15_48_955"] # Split and noise alternate
#runs = ["LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59"] # pick step 50k of this one - results without ghosts (might work better on parton-level)

#runs = ["IRC_loss_Split_and_Noise_alternate_Aug_2025_04_14_11_10_21_788", "IRC_loss_Split_and_Noise_alternate_NoAug_2025_04_11_16_15_48_955"] # pick step 12900 of these


runs = [["LGATr_training_NoPID_Delphes_10_16_64_0.8_2025_04_17_18_07_38_405", 60000],
        ["LGATr_500part_NOQMin_Delphes_2025_04_19_11_15_24_417", 14820],
        ["Delphes_IRC_aug_SplitOnly_2025_04_20_15_50_33_553", 14280]
        ] # step 60k: initial clustering training on Delphes 900_03 dataset

#runs = ["LGATr_500part_NOQMin_Delphes_2025_04_19_11_15_24_417"]#, "Delphes_IRC_aug_2025_04_19_11_16_17_130"]#step 12660

#runs = ["Delphes_IRC_aug_SplitOnly_2025_04_20_15_50_33_553"] # IRC loss only with splitting
#runs = ["Delphes_IRC_NOAug_SplitOnly_2025_04_21_12_58_36_99", "Delphes_IRC_NOAug_SplitAndNoise_2025_04_21_19_32_08_865"]
#runs = ["CONT_Delphes_IRC_aug_SplitOnly_2025_04_21_12_53_27_730"]



# PFfix Delphes runs
runs = [
    ["LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 2000],
    ["LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 4000],
    ["LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 10000],
    ["LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 20000],
    ["LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 40000],
    ["LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 50000],
]


runs = [
    #["Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755", 2000],
    #["Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755", 4000],
    #["Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755", 10000],
    #["Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755", 20000],
    ["Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755", 40000],
    ["Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755", 50000],
    ["Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755", 60000],
]

'''runs = [
    ["GATr_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_35_48_163", 2000],
    ["GATr_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_35_48_163", 4000],
    ["GATr_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_35_48_163", 10000],
    ["GATr_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_35_48_163", 20000],
    ["GATr_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_35_48_163", 40000],
    ["GATr_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_35_48_163", 50000],
]
runs = [["LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134", 60000]]
'''

test_files = ["PFNano_s-channel_mMed-1000_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
              "PFNano_s-channel_mMed-1000_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
              "PFNano_s-channel_mMed-1000_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
              "PFNano_s-channel_mMed-1100_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-1100_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-1100_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-1200_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-1200_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-700_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-800_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-800_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-800_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-900_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
                "PFNano_s-channel_mMed-900_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000"
              ]

test_files_smaller = ["PFNano_s-channel_mMed-1000_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1000_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1000_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1100_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1100_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1100_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1200_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-1200_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-700_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000",
"PFNano_s-channel_mMed-800_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"]

test_files_delphes = ['SVJ_mZprime-1100_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-700_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1500_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-1000_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1200_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-1400_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1100_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1300_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-700_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-800_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-1500_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1000_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-1400_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-1500_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-800_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1300_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1100_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-1200_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-900_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-1300_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-700_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-800_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-1400_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-900_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1200_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1000_mDark-20_rinv-0.3_alpha-peak']
test_files_delphes = 'SVJ_mZprime-1000_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-1000_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1000_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-1100_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-1100_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1100_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-1200_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-1200_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-1200_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-700_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-700_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-700_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-800_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-800_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-800_mDark-20_rinv-0.7_alpha-peak', 'SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak', 'SVJ_mZprime-900_mDark-20_rinv-0.5_alpha-peak', 'SVJ_mZprime-900_mDark-20_rinv-0.7_alpha-peak'
#test_files = ["Feb26_2025_E1000_N500_noPartonFilter_C_F/" + x for x in test_files]
#test_files = ["Feb26_2025_E1000_N500_noPartonFilter_GluonFixF/" + x for x in test_files]
test_files = ["Feb26_2025_E1000_N500_noPartonFilter_GluonFix_Small2K_F_part0/" + x for x in test_files]
#test_files = ["Feb26_2025_E1000_N500_folders/" + x for x in test_files_smaller]
test_files = ["Delphes_020425_test_PU_PFfix_part0/" + x for x in test_files_delphes] # Delphes test files
print("------")

for run, step in runs:
    #for level in ["-pl", ""]:
    #for level in ["-pl", ""]:
    for level in ["-gl", "", "-pl"]:
    #for level in ["-pl"]:
    #for level in ["-gl", "-pl", ""]:
    #for level in [""]:
        #for aug_suffix in ["-aug-soft"]:
        #aug_suffixes = ["-aug-soft"]
        #if step >= 40000:
        #    aug_suffixes = [""] # without ghosts for the base clustering
        aug_suffixes = [""]
        for aug_suffix in aug_suffixes:
            print("python -m scripts.generate_test_jobs -run {} -step {} --steps-from-zero -template t3 -tag DelphesPFfix {} --custom-test-files \"{}\" {} ".format(run, step, level, " ".join(test_files), aug_suffix))

print("-----")
