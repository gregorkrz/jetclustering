# SVJ clustering
The repo has evolved from [here](https://github.com/selvaggi/mlpf) - mainly, we use the dataloader and code for reading  the root files for the previous MLPF project

## Setup
**Important**: To make it easier and less time-consuming to move the commands across different machines, i.e. lxplus, T3 and Vega, we use relative paths. However, all commands can also be supplied absolute paths starting with `/`. **In case you use relative paths, make sure to modify the `env.sh` file with your paths!** 
1. Set the environment variables `source env.sh`


### Preprocess data

#### First version without dark quarks
`python -m src.preprocessing.preprocess_dataset --input SVJ_std_UL2018_scouting_test_large --output SVJ_std_UL2018_scouting_test_large`

### Training models

### Evaluation

