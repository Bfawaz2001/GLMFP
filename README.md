# Generative Language Model for Proteins (GLMFP)

## Introduction
The GLMFP project develops a Command Line Interface (CLI) that implements various language models to generate realistic synthetic protein sequences. These models include N-gram, LSTM, and Transformer models, all trained using the Reviewed Swiss-Prot database. This project aims to provide a user-friendly interface that incorporates all functionalities for generating and analyzing proteins in one seamless program.

## Environment Setup

### Dependencies
- Python 3.9.18
- Diamond 2.1.9
- PyTorch 2.2.0
- BioPython 1.78
- NumPy 1.26.4
- Pandas 2.1.4
- Scikit-learn 1.3.2

### Virtual Environment
To avoid conflicts with other Python projects, it is recommended to use a virtual environment:

```bash
# Create a virtual environment
conda create -n glmfp_env python=3.9.18

# Activate the environment
conda activate glmfp_env

# Install required packages
conda install biopython
conda install numpy
conda install pandas
conda install scikit-learn
pip install torch==2.2.0
```
## Running the Models on an HPC System

### Prerequisites
Before running the models on a High-Performance Computing (HPC) system, ensure that your NVIDIA GPU is compatible and that the correct version of CUDA is installed to match the GPU capabilities and requirements.

### Setting Up SLURM Batch Scripts
To submit jobs to an HPC cluster, SLURM (Simple Linux Utility for Resource Management) is used. Below is a template for a typical SLURM batch script which requests a GPU resource, specifies job time, and allocates memory.

#### SLURM Script Example
Here is an example of a SLURM script that you might use to run a neural network or transformer model on the HPC:

```bash
#!/bin/bash
#SBATCH --job-name=glmfp   # Job name
#SBATCH --gres=gpu:1       # Request GPU resource
#SBATCH --time=80:00:00    # Time limit hrs:min:sec
#SBATCH --mem=200G          # Memory needed per node
#SBATCH --output=glmfp_%j.log  # Standard output and error log

# Load necessary modules
module load cuda/11.2

# Activate the virtual environment
source activate glmfp_env

# Execute the Python script
python path/to/your_neural_netowrk_or_transformer_build_script.py
```
### Submitting a Job
Save this script as run_glmfp.sh. You can submit the job to the HPC system by using the following SLURM command:

```bash
sbatch run_glmfp.sh
```

This command queues the job in the HPC system, and it will execute once the requested resources become available.

### Monitoring Job Status
After submitting the job, you can check the status of your job using:

```bash
squeue 
# or 
sacct
```
The squeue command provides a list of all queued and running jobs, including your own, displaying their current status 
on the HPC system. The sacct command displays only your submitted job status and run time.

### Job Output
The output from your job, including any print statements and errors, will be captured in the file specified in the SLURM
script (glmfp_%j.log). The %j in the file name will be replaced by the job ID, allowing you to easily identify and check
the output for your specific job.

## Building the Models

### Structure of Model Building Scripts
The model building scripts are organized into separate folders under the `./src/build_models` directory. Each model type 
has its dedicated folder and script, facilitating easy navigation and maintenance.

#### N-gram Models
In the `n gram models` folder, there are multiple scripts, each corresponding to a different N-gram model (e.g., 
`2mer_model.py`, `3mer_model.py`, etc.). These scripts are used to train models on sequences to predict the likelihood 
of each amino acid based on the previous `N-1` amino acids. The building process involves calculating the frequency of 
each N-gram from the training dataset and using these frequencies to estimate the probabilities of amino acid sequences.

#### Neural Network Models
The `neural network models` folder contains scripts like `nn_lstm.py`. This script builds a Long Short-Term Memory 
(LSTM) model that is capable of learning long-term dependencies in sequence data. The LSTM model uses layers of memory 
cells to process data sequentially and retain information across long sequences, making it ideal for protein sequence 
generation.

#### Transformer Models
The `transformer_model` folder includes `transformer.py`, which implements a Transformer model. Unlike LSTM, the 
Transformer model uses self-attention mechanisms to weigh the importance of different words in the sequence without 
regard to their input order, providing a more flexible and powerful way to model sequence data. This model is 
particularly effective at capturing complex dependencies and patterns in protein sequences.

All neural network and Transformer models utilise GPU acceleration via CUDA, ensuring efficient computation and faster 
training times on compatible hardware.

### Running The Build Scripts

Build scripts are run using the following command:
``` bash 
# Activate the environment
conda activate glmfp_env

# Run the build script
python chosen_model.py
```

## Main Program: `glmfp_main_program.py`

### Program Structure
The `glmfp_main_program.py` script is divided into two main parts:

1. **Generating Sequences:** This part allows the user to generate new protein sequences using any of the trained models. 
Users can specify the model type, the number of sequences, and the length of sequences to generate.

2. **Analyzing Sequences:** After generating sequences, users can analyze them using various built-in tools to assess their properties and biological relevance.

### User Parameters for Generating Proteins
Users are prompted to input several parameters when generating proteins:

- **Model Selection:** Choose from N-gram, LSTM, or Transformer models.
- **Number of Proteins:** Specify how many proteins to generate.
- **Length Range:** Define the minimum and maximum lengths of the proteins.

### Output of Generated Sequences
The generated sequences using the models are output as a `.fasta` file in `./results/generated proteins` directory with 
the model name as a prefix automatically added. 

### Analysis Tools
The program includes several tools for analyzing the generated protein sequences:

- **DIAMOND BLASTp:** Compares generated sequences against a large reference database of proteins to find matches and assess the novelty of the sequences.
- **InterProScan:** Provides functional analysis of proteins by classifying them into families and predicting domains and important sites.
- **Analysis Summary:** Calculates and displays sequence specific properties including amino acid composition, Shannon entropy, and physicochemical properties.

### Output Formats and Directories Of Analysis Tools

#### DIAMOND BLASTp
- **Directory:** `./results/diamond blastp results/`
- **Output Format:** The output files are in `XML` for search results and `.txt` for the summary, containing detailed 
alignment information such as query ID, subject ID, percentage of identity, alignment length, mismatches, gap opens, 
q.start, q.end, s.start, s.end, evalue, and bit score. 
- **Usage:** This tool compares generated protein sequences against known protein databases (e.g., NCBI nr) to assess similarity and potential functionality.

#### InterProScan
- **Directory:** `./results/interpro results/`
- **Output Format:** Results are saved in `.json` format and summary in `.txt` file, providing comprehensive functional 
analysis of protein sequences. This includes annotations for protein families, domains, sites, and pathways, helping users understand the biological 
implications of their synthetic sequences.
- **Usage:** Functional annotations are crucial for determining the roles and characteristics of the proteins, aiding in further biological or computational research.

#### Summary Text File
- **File Name:** `{filename}_summary.txt`
- **Location:** Inside the respective analysis folder (e.g., `./results/analysis_summary/{selected_fasta_file}/`)
- **Content:** This text file includes detailed metrics for each protein in the input file:
  - Amino acid counts and frequencies for each protein.
  - Shannon Entropy, indicating the diversity of amino acid usage.
  - Molecular Weight, providing the sum of the molecular weights of the amino acids in the protein.
  - Isoelectric Point (pI), which is the pH at which the protein carries no net electrical charge.
- **End of File:** The file concludes with an overall amino acid composition analysis of all proteins combined, giving a holistic view of the batch's compositional makeup.

#### Amino Acid Composition Graphs
- **Folder Name:** `aa_composition_graphs`
- **Location:** Within each analysis-specific folder (e.g., `./results/analysis_summary/{selected_fasta_file}/aa_composition_graphs/`)
- **Content:** This folder contains:
  - Individual graph files (in PNG format) for each protein, showing the amino acid composition.
  - A combined graph file titled `total_aa_composition.png`, which represents the amino acid composition across all analyzed proteins, allowing for comparative and aggregate views.

### Running The Program

The models can generate proteins based on user input through a CLI:
```bash 
# Activate the environment
conda activate glmfp_env

# Run the main program
python glmfp_main_program.py
```
Menus will display options and prompt the user allowing a navigation throughout the program.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements
Thanks to the IBERS department for HPC access, and special thanks to all project contributors and supervising staff.