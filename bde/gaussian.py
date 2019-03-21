import os
import tempfile
import subprocess
import logging
import uuid
import time

import numpy as np
import cclib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetBondLength

logging.getLogger("cclib").setLevel(30)


class GaussianRunner(object):

    def __init__(self, smiles, cid, type_, max_conformers=1000,
                 min_conformers=100, nprocs=18, mem='40GB',
                 scratchdir='/tmp/scratch',
                 projectdir='/projects/cooptimasoot/psj_bde/'):
        """ Class to handle the overall temporary directory management for
        running Gaussian on Eagle """

        self.smiles = smiles
        self.cid = cid
        self.type_ = type_
        self.max_conformers = max_conformers
        self.min_conformers = min_conformers
        self.nprocs = nprocs
        self.mem = mem
        self.scratchdir = scratchdir
        self.projectdir = projectdir


    def process(self):

        with tempfile.TemporaryDirectory(dir=self.scratchdir) as tmpdirname:

            print("starting SMILES {}".format(self.smiles))
            mol, confId = self.optimize_molecule_mmff()
            self.write_gaussian_input_file(mol, confId, tmpdirname)

            # Run gaussian and time calculation
            gauss_start_time = time.time()
            self.run_gaussian(tmpdirname)
            gauss_run_time = time.time() - gauss_start_time
            print("python walltime for SMILES {0}: {1}".format(
                self.smiles, gauss_run_time))


            mol, enthalpy = self.parse_log_file()
            log = self.cleanup()
            
            molstr = Chem.MolToMolBlock(mol)

            return molstr, enthalpy, log


    def optimize_molecule_mmff(self):
        """ Embed a molecule in 3D space, optimizing a number of conformers and
        selecting the most stable
        """
        
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.rdmolops.AddHs(mol)

        # Use min < 3^n < max conformers, where n is the number of rotatable bonds
        NumRotatableBonds = AllChem.CalcNumRotatableBonds(mol)
        NumConformers = np.clip(3**NumRotatableBonds, self.min_conformers,
                                self.max_conformers)

        conformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=int(NumConformers), pruneRmsThresh=0.2, randomSeed=1,
            useExpTorsionAnglePrefs=True, useBasicKnowledge=True)

        def optimize_conformer(conformer):
            prop = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=conformer)
            ff.Minimize()
            return float(ff.CalcEnergy())

        assert conformers, "Conformer embedding failed"

        if len(conformers) == 1:
            logging.critical(
                'Only 1 conformer for SMILES {}'.format(self.smiles))
            most_stable_conformer = conformers[0]
            
        else:
            conformer_energies = np.array(
                [optimize_conformer(conformer) for conformer in conformers])
            most_stable_conformer = conformer_energies.argmin()

        return mol, int(most_stable_conformer)


    def write_gaussian_input_file(self, mol, confId, tmpdirname):
        """ Given an rdkit.Mol object with an optimized, minimum energy conformer
        ID, write a gaussian input file using openbabel to the scratch folder """

        self.run_hex = uuid.uuid4().hex[:6]
        self.input_file = tmpdirname + '/{0}_{1}.gjf'.format(self.cid, self.run_hex)
        checkpoint_file = tmpdirname + '/{0}_{1}.chk'.format(self.cid, self.run_hex)

        with tempfile.NamedTemporaryFile(
                'w', suffix='.sdf', dir=tmpdirname) as sdf_file:
            writer = Chem.SDWriter(sdf_file)
            mol.SetProp('_Name', str(self.cid))
            writer.write(mol, confId=confId)
            writer.close()

            if self.type_ is 'fragment':
                # Run stable=opt
            
                header1 = [
                    '%chk={0}'.format(checkpoint_file),        
                    '%MEM={}'.format(self.mem),
                    '%nprocshared={}'.format(self.nprocs),
                    '# stable=opt M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400)'
                    ' nosymm guess=mix']
                
                subprocess.call(
                    ['obabel', sdf_file.name, '-O', input_file, '-xk',
                     '\n'.join(header1)])

                    
                with open(input_file, 'r') as f:
                    chg_mul = f.readlines()[7]

                with open(input_file, 'a') as f:
                    
                    header2 = [
                        '--link1--',
                        '%chk={0}'.format(checkpoint_file),        
                        '%MEM={}'.format(self.mem),
                        '%nprocshared={}'.format(self.nprocs),
                        '# opt freq M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400)'
                            ' nosymm guess=read geom=check\n',
                        ' {}\n'.format(mol.GetProp('_Name')),
                        chg_mul
                    ]
                    
                    f.write('\n'.join(header2))

            else:

                header1 = [
                    '%MEM={}'.format(self.mem),
                    '%nprocshared={}'.format(self.nprocs),
                    '# opt freq M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400) nosymm']

                subprocess.call(
                    ['obabel', sdf_file.name, '-O', input_file, '-xk', '\n'.join(header1)])
        

    def run_gaussian(self, tmpdirname):
        """ Run the given Guassian input file (with associated mol ID) """

        self.log = tmpdirname + '{0}_{1}.log'.format(self.cid, self.run_hex)
        
        with tempfile.TemporaryDirectory(dir=tmpdirname) as gausstmp:
            env = os.environ.copy()
            env['GAUSS_SCRDIR'] = gausstmp
            subprocess.call("module load gaussian/G16B && g16 < {0} > {1}".format(
                self.gjf, self.log), shell=True, env=env)


    def parse_log_file(self):
        """ Parse the gaussian log file using cclib, return the optimized mol and
        enthalpy. """

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
        
        covalent_radii = {'H': .31, 'C': .76, 'N': .71, 'O': .66}
        
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

        return mol, data.enthalpy

    def cleanup(self):
        """ Compress files and store in /projects """

        log_basename = os.path.basename(self.log)
        gjf_basename = os.path.basename(self.gjf)

        newlog = self.projectdir + log_basename + '.gz'
        newgjf = self.projectdir + gjf_basename + '.gz'

        subprocess.call(['gzip', self.log, gjf])
        subprocess.call(['mv', self.log + '.gz', newlog])
        subprocess.call(['mv', self.gjf + '.gz', newgjf])

        return newlog
