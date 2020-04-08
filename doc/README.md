# Example files for creating and loading the QM database

In this directory, I've included two additional files that were used to create the QM database. These scripts are highly dependent on my local HPC architecture, and will likely need to be modified heavily to work with a different HPC resource or database.
* `gaussian_worker.py`: a python script that interacts with a previously created PostgresQL database and the `gaussian.py` script in `bde/` to conduct the optimizations and validations described in the manuscript. 
* `batch_submit.sh`: a SLURM scheduler script that batches the `gaussian_worker.py` script across multiple compute nodes.

In `sdf_load_demonstration_with_rdkit.ipynb`, I demonstrate how to use RDKit (2019.09.3) to load the SDF file, decompose molecules into radical fragments for BDE calculations (using `bde/fragment.py`), and use the database to calculate BDE values.