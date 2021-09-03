import os
import tempfile
import subprocess
import logging
import uuid
import time
import socket

import numpy as np
import cclib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetBondLength

logging.getLogger("cclib").setLevel(30)


class GaussianRedoxRunner(object):

    def __init__(self, smiles, cid, mol, type_, max_conformers=1000,
                 min_conformers=100, nprocs=18, mem='40GB',
                 scratchdir='/tmp/scratch',
                 projectdir='/projects/rlmolecule/pstjohn/redox_calculations/',
                 gaussian_timeout=86400):
        """ Class to handle the overall temporary directory management for
        running Gaussian on Eagle """

        self.smiles = smiles
        self.cid = cid
        self.mol = mol
        self.type_ = type_
        self.nprocs = nprocs
        self.mem = mem
        self.scratchdir = scratchdir
        self.projectdir = projectdir
        self.gaussian_timeout = gaussian_timeout


    def process(self):

        with tempfile.TemporaryDirectory(dir=self.scratchdir) as tmpdirname:

            print("starting SMILES {0}, {1} on host {2}".format(
                self.smiles, self.type_, socket.gethostname()))
            self.write_gaussian_input_file(self.mol, tmpdirname)

            # Run gaussian and time calculation
            gauss_start_time = time.time()
            self.run_gaussian(tmpdirname)
            gauss_run_time = time.time() - gauss_start_time
            print("python walltime for SMILES {0} on host {1}: {2}".format(
                self.smiles, socket.gethostname(), gauss_run_time))


            mol, enthalpy, freeenergy, scfenergy = self.parse_log_file()
            log = self.cleanup()
            
            molstr = Chem.MolToMolBlock(mol)

            return molstr, enthalpy, freeenergy, scfenergy, log


    def write_gaussian_input_file(self, mol, tmpdirname):
        """ Given an rdkit.Mol object with an optimized, minimum energy conformer
        ID, write a gaussian input file using openbabel to the scratch folder """

        self.run_hex = uuid.uuid4().hex[:6]
        self.gjf = tmpdirname + '/{0}_{1}.gjf'.format(self.cid, self.run_hex)
        checkpoint_file = tmpdirname + '/{0}_{1}.chk'.format(self.cid, self.run_hex)

        with tempfile.NamedTemporaryFile(
                'wt', suffix='.sdf', dir=tmpdirname) as sdf_file:
            sdf_file.write(mol)
            sdf_file.flush()            

            header1 = [
                '%chk={0}'.format(checkpoint_file),
                '%MEM={}'.format(self.mem),
                '%nprocshared={}'.format(self.nprocs),
                '# stable=opt M062X/Def2TZVP scrf=(SMD,solvent=water)'
                ' nosymm guess=mix']

            subprocess.run(
                ['obabel', sdf_file.name, '-O', f'{tmpdirname}/temp_{self.run_hex}.gjf', '-xk',
                 '\n'.join(header1)])

        with open(f'{tmpdirname}/temp_{self.run_hex}.gjf', 'rt') as f:
            inputs = f.read().split('\n')
            inputs[5] = f' {self.cid}'
            chg_mul = inputs[7]

            #rewriting the charges for the oxidised and reduced species.
            if self.type_ =='oxidized' or self.type_ =='reduced':
                if self.type_ =='oxidized':
                	chg = int(chg_mul.split()[0]) + 1
                	mul = int(chg_mul.split()[1]) - 1
                elif self.type_ =='reduced':
                	chg = int(chg_mul.split()[0]) - 1
                	mul = int(chg_mul.split()[1]) - 1                    
                    
                chg_mul = f'{chg} {mul}'
                inputs[7] = chg_mul
            
            inputs += [
                '--link1--',
                '%chk={0}'.format(checkpoint_file),
                '%MEM={}'.format(self.mem),
                '%nprocshared={}'.format(self.nprocs),
                '# opt freq M062X/Def2TZVP scrf=(SMD,solvent=water)'
                ' nosymm guess=read geom=check\n',
                ' {}\n'.format(self.cid),
                chg_mul
            ]            
            
        with open(self.gjf, 'wt') as f:
            f.write('\n'.join(inputs))
            
        # debug -- keep a copy before running gaussian
        gjf_basename = os.path.basename(self.gjf)
        newgjf = self.projectdir + 'gjf_errors/' + gjf_basename
        subprocess.run(['cp', self.gjf, newgjf])
        

    def run_gaussian(self, tmpdirname):
        """ Run the given Guassian input file (with associated mol ID) """

        self.log = tmpdirname + '/{0}_{1}.log'.format(self.cid, self.run_hex)
        self.log = os.path.join(self.projectdir, 'log', '{0}_{1}.log'.format(self.cid, self.run_hex))

        gaussian_cmd = "module load gaussian/G16C && g16 < {0} > {1}".format(
            self.gjf, self.log)
        
        with tempfile.TemporaryDirectory(dir=tmpdirname) as gausstmp:
            env = os.environ.copy()
            env['GAUSS_SCRDIR'] = gausstmp
            subprocess.run(gaussian_cmd, shell=True, env=env,
                           timeout=self.gaussian_timeout)


    def parse_log_file(self):
        """ Parse the gaussian log file using cclib, return the optimized mol and
        enthalpy. """
        
        # Debug, store log before parsing
        log_basename = os.path.basename(self.log)
        newlog = os.path.join(self.projectdir, 'log_errors/', log_basename)
        subprocess.run(['cp', self.log, newlog])                

        # Parse the output log with cclib, assert the optimization completed
        data = cclib.io.ccread(self.log)
        assert data.optdone, "Optimization not converged"
        
        if hasattr(data, 'vibfreqs'): # single atoms don't have this property
            assert min(data.vibfreqs) >= 0, "Imaginary Frequency"

        # Create an RDKit Molecule from the SMILES string
        mol = Chem.MolFromSmiles(self.smiles)
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()
        
        assert np.allclose(
            np.array([a.GetAtomicNum() for a in mol.GetAtoms()]),
            data.atomnos), "Stoichiometry check failed"

        # Correct the 3D positions of the atoms using the optimized geometry
        for i in range(conf.GetNumAtoms()):
            conf.SetAtomPosition(i, data.atomcoords[-1][i])
        
        covalent_radii = {'H': .31, 'C': .76, 'N': .71,
                          'O': .66, 'P': 1.07, 'S': 1.05,
                          'F': .57, 'Cl': 1.02, 'Br': 1.20}

        # Check bond lengths
        for bond in mol.GetBonds():
            length = GetBondLength(
                mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
            max_length = (covalent_radii[bond.GetBeginAtom().GetSymbol()] + 
                          covalent_radii[bond.GetEndAtom().GetSymbol()] + 0.4)
            
            assert length <= max_length, "bond greater than maximum covalent length"


        # Set property fields of the molecule for final SDF export
        mol.SetProp("_Name", str(self.cid))
        mol.SetProp('SMILES', self.smiles)
        mol.SetDoubleProp('Enthalpy', data.enthalpy)

        return mol, data.enthalpy, data.freeenergy, data.scfenergies[-1] / 27.2114

    def cleanup(self):
        """ Compress files and store in /projects """

        log_basename = os.path.basename(self.log)
        gjf_basename = os.path.basename(self.gjf)

        newlog = os.path.join(self.projectdir, 'log/', log_basename + '.gz')
        newgjf = os.path.join(self.projectdir, 'gjf/', gjf_basename + '.gz')

        subprocess.run(['gzip', self.log, self.gjf])
        subprocess.run(['mv', self.log + '.gz', newlog])
        subprocess.run(['mv', self.gjf + '.gz', newgjf])
        
        # Remove debugging files
        subprocess.run(['rm', os.path.join(self.projectdir, 'gjf_errors/', gjf_basename)])
        subprocess.run(['rm', os.path.join(self.projectdir, 'log_errors/', log_basename)])        
        
        return newlog
