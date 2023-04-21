# Define aliases to manage the experiments and runs

# Quick initialization of new run. Usage: init -exp_name [exp. name] -run_name [run name] -conf [config file name]
alias init = 'python management.py --init_run --conf'

# Initialize experiment. Usage: iexp -exp_name [exp. name]
alias iexp = 'python management.py --init_exp'

# Initialize run. Usage: irun -exp_name [exp. name] -run_name [run name]
alias irun = 'python management.py --init_run'

# Clear run and reset to initialization state. Usage: clr -exp_name [exp. name] -run_name [run name]
alias clr = 'python management.py --clear_run'

# Load config from the ./config directory to a run directory. Usage: conf -exp_name [exp. name] -run_name [run name] -conf [config file name]
alias conf = 'python management.py --conf'