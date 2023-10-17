# Llama2-Inference

This repository contains example code for performing inference with Llama 2 using 🤗 Transformers on a supercomputer.

The process consists of the following steps:
1. Create an Apptainer file using the tested `llama2_2.def` and `mamba_llama2_2.yml` files.
2. Develop a Python script using the `llama2.ipynb` Jupyter Notebook. Jupyter Notebooks are a favourite among data scientists!
3. Create a Python as `llama2.py`. My script is quite a mess as I worked alone. No one will adjust it.
4. Create Slurm scripts `7b_prompt-1.sh` and `13b_prompt-1.sh` to run the Python script that we created.
5. Submit the Slurm job to execute it.
