import itertools
import subprocess

# Define the models and datasets
models = ["tpvformer"]
datasets = ["kitti360", "occ3d", "surroundocc"]

# Base command template with default cfg-options
base_command = "python train.py --py-config config/{model}/render.py --work-dir out/debug --dataset {dataset} --cfg-options inspect=True"

# Additional configuration options for specific datasets
additional_cfg_options = (
    "train_dataset_config.is_mini=True val_dataset_config.is_mini=True"
)


# Function to construct the command
def construct_command(model, dataset):
    command = base_command.format(model=model, dataset=dataset)
    if dataset != "kitti360":
        command += f" {additional_cfg_options}"
    return command


# Output file
output_file = "./tests/run_results.txt"

# Iterate over all combinations of models and datasets
with open(output_file, "w") as f:
    for model, dataset in itertools.product(models, datasets):
        command = construct_command(model, dataset)
        print(f"Running command: {command}")
        f.write(f"Running command: {command}\n")

        # Run the command and capture the output
        result = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Check if the command was successful
        if result.returncode == 0:
            f.write(f"Command succeeded: {command}\n\n")
        else:
            f.write(f"Command failed: {command}\n")
            f.write(f"Error: {result.stderr.decode()}\n\n")

print(f"Results have been written to {output_file}")
