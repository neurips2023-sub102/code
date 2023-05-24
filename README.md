
## Code Organization
The Bayesian optimization loop is in `main.py`

`models`: the model code for each of the surrogate models we consider.

`test_functions`: objective functions for benchmark problems



## Installation
Create a new conda environment:
````
conda env create -f environment.yml -n <env_name>
````

Install the project:
````
pip install -e .
````

## Running experiments
Each experiment requires a config json, and there are examples of in `config`. 

To use the config file `<name>.json`, run the following command from the root folder
````
python main.py --config <name>
````

You can also include the `--bg` flag if you would like to redirect stderr and stdout to a different file and save it.
````
python main.py --config <name> --bg
````

