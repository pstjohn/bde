# Utilities for radical enthalpy and bond dissociation calculations

This repository contains helper functions used in the creation and analysis of the radical database discussed in "Quantum chemical calculations for over 200,000 organic radicals and 40,000 associated closed-shell molecules." by Peter St John, Yanfei Guan, Yeonjoon Kim, Brian Etz, Seonah Kim, and Robert Paton, currently under review at *Scientific Data*.

Organization of the repository is as follows:

`bde/`: 
* `fragment.py`: contains the code used to break single, non-ring bonds in input molecules in order to yield monovalent radicals, and return a dataframe indicating the bond index of the breaking bond and canonicalized SMILES representations of the product radicals.
* `gaussian.py`: contains code to take input SMILES strings, calculate lowest-energy conformers, create the input Gaussian file, run the gaussian calculation, and parse the resulting logfile.

`doc/`:
* `gaussian_worker.py`: a python script that interacts with a previously created PostgresQL database and the `gaussian.py` script in `bde/` to conduct the optimizations and validations described in the manuscript. 
* `batch_submit.sh`: a SLURM scheduler script that batches the `gaussian_worker.py` script across multiple compute nodes.
* `sdf_load_demonstration_with_rdkit.ipynb`: a jupyter notebook that shows how RDKit can be used to parse the SDF file (distributed via Figshare) and calculate bond dissociation energies
* `count_radical_types.ipynb`: a jupyter notebook that loads the SDF file and counts the formal radical centers by location of the unpaired electron on primary/secondary/tertiary C/N/O atom, as well as by characterizing the radical by nearby substituents.