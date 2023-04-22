# Define aliases to manage the experiments and runs

# Environemnt variables for the experiment manager

# Quick initialization of new run. Usage: init -exp_name [exp. name] -run_name [run name] -conf [config file name]
alias init='python $EXP_DIRECTORY/utils/management.py --init_run --conf'

# Initialize experiment. Usage: iexp -exp_name [exp. name]
alias iexp='python $EXP_DIRECTORY/utils/management.py --init_exp'

# Initialize run. Usage: irun -exp_name [exp. name] -run_name [run name]
alias irun='python $EXP_DIRECTORY/utils/management.py --init_run'

# Clear run and reset to initialization state. Usage: clr -exp_name [exp. name] -run_name [run name]
alias clr='python $EXP_DIRECTORY/utils/management.py --clear_run'

# Load config from the ./config directory to a run directory. Usage: conf -exp_name [exp. name] -run_name [run name] -conf [config file name]
alias conf='python $EXP_DIRECTORY/utils/management.py --conf'

alias del='python $EXP_DIRECTORY/utils/management.py --clear_data'

# Activate experiment or run for easier running of scripts. Usage
function setexp() {
    export ACTIVATE_EXP="$1"
}
function setrun() {
    export ACTIVATE_RUN="$1"
}
