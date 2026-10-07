# Syngenic Channels

## Hypothesis
LLMs share meaning representations for unknown words across models. 

## Setup Environment (once)
Replace `<PROJ_DIR>` with the path to the project. The environment (`<PROJ_ENV>`) is set to `syngeny-env` by default.
``
    $ cd <PROJ_DIR>
    $ ./bash/setup_env.sh <PROJ_DIR> [<PROJ_ENV>]
`` 

## Load environment
``
    $ conda activate <PROJ_ENV>
``

## Data Processing
``
    $ ./bash/scripts/process_data.sh
``
## Run experiments

To perform the experiments end-to-end, run:
``
    $ ./bash/scripts/run_experiments.sh
``
