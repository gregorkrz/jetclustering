

for pt in [30, 40, 50, 60, 70, 80, 90]:
    #for suffix in ["", "-pl", "-gl"]:
    for suffix in [""]:
        #cmd = f"python -m scripts.test_plot_jobs --tag DelphesPFfix_FullDataset_TrainDSstudy_QCD --input QCD_test_part0 --submit-AKX -pt {pt}  {suffix}"
        cmd = f"python -m scripts.test_plot_jobs --tag DelphesPFfix_FullDataset_TrainDSstudy --input Delphes_020425_test_PU_PFfix_part0 --submit-AKX -pt {pt}  {suffix}"
        print(cmd)

'''
for pt in [30, 40, 50, 60, 70, 80, 90]:
    #cmd = f"python -m scripts.test_plot_jobs --tag DelphesPFfix_FullDataset_TrainDSstudy --input Delphes_020425_test_PU_PFfix_part0 -pt {pt}"
    #print(cmd)
    cmd = f"/work/gkrzmanc/1gatr/bin/python -m scripts.plot_eval_count_matched_quarks --input QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_{str(int(pt))}.0"
    #cmd = f"/work/gkrzmanc/1gatr/bin/python -m scripts.plot_eval_count_matched_quarks --input Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_{str(int(pt))}.0"
    print(cmd)

'''