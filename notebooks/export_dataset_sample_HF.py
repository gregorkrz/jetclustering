# Export a couple of events for each dataset into txt files for demo
from pathlib import Path
import os
from src.dataset.dataset import EventDatasetCollection, EventDataset

from src.utils.paths import get_path

n_events_per_dataset = 50

inputs = {
    "QCD": "QCD_test_part0/qcd_test"
}

for rinv in [0.3, 0.5, 0.7]:
    for mmed in [700, 800, 900, 1000, 1100, 1200]:
        inputs[f"r_inv.={rinv}, m_med.={mmed} GeV"] = f"Delphes_020425_test_PU_PFfix_part0/SVJ_mZprime-{mmed}_mDark-20_rinv-{rinv}_alpha-peak"

print(inputs)

Path("demo_datasets").mkdir(exist_ok=True)
for key in inputs:
    Path(os.path.join("demo_datasets", key)).mkdir(exist_ok=True)
    dataset = EventDataset.from_directory(get_path(inputs[key], "preprocessed_data"), mmap=True)
    for n in range(n_events_per_dataset):
        event = dataset[n]
        pfcands_out = ""
        for i in range(len(event.pfcands)):
            pfcands_out += f"{event.pfcands.pt[i].item()} {event.pfcands.eta[i].item()} {event.pfcands.phi[i].item()} {event.pfcands.mass[i].item()} {event.pfcands.charge[i].item()}\n"
        gen_particles_out = ""
        for i in range(len(event.matrix_element_gen_particles)):
            gen_particles_out += f"{event.matrix_element_gen_particles.pt[i].item()} {event.matrix_element_gen_particles.eta[i].item()} {event.matrix_element_gen_particles.phi[i].item()}\n"
        # write the pfcands_out and gen_particles_out to os.path.join("demo_datasets", key, "n".txt) and n_quarks.txt
        with open(os.path.join("demo_datasets", key, str(n)+".txt"), "w") as f:
            f.write(pfcands_out)
        with open(os.path.join("demo_datasets", key, str(n)+"_quarks.txt"), "w") as f:
            f.write(gen_particles_out)

