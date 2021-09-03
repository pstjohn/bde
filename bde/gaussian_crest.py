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
from rdkit.Chem import PeriodicTable
from rdkit.Chem.rdMolTransforms import GetBondLength

logging.getLogger("cclib").setLevel(30)


class GaussianRunner(object):

    def __init__(self, smiles, cid, type_, max_conformers=1000,
                 min_conformers=100, nprocs=18, mem='40GB',
                 scratchdir='/tmp/scratch',
                 projectdir='/projects/cooptimasoot/psj_bde/',
                 crest_timeout=86400,
                 gaussian_timeout=86400):
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
        self.crest_timeout = crest_timeout
        self.gaussian_timeout = gaussian_timeout


    def process(self):

        with tempfile.TemporaryDirectory(dir=self.scratchdir) as tmpdirname:

            print("starting SMILES {0} on host {1}".format(self.smiles, socket.gethostname()))
            mol, confId = self.optimize_molecule_mmff()

            # running crest
            self.run_crest(mol, confId,tmpdirname)

            self.write_gaussian_input_file(tmpdirname)

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


    def optimize_molecule_mmff(self):
        """ Embed a molecule in 3D space, optimizing a number of conformers and
        selecting the most stable
        """

        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.rdmolops.AddHs(mol)

        # If the molecule is a radical; add a hydrogen so MMFF converges to a
        # reasonable structure
        is_radical = False
        radical_index = None
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetNumRadicalElectrons() != 0:
                is_radical = True
                radical_index = i

                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                atom.SetNumRadicalElectrons(0)


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

        # If hydrogen was added; remove it before returning the final mol
        if is_radical:
            radical_atom = mol.GetAtomWithIdx(radical_index)
            radical_atom.SetNumExplicitHs(int(radical_atom.GetNumExplicitHs()) - 1)
            radical_atom.SetNumRadicalElectrons(1)

        return mol, int(most_stable_conformer)


    def run_crest(self, mol, confId, tmpdirname):
        """ Given an rdkit.Mol object with an optimized, minimum energy conformer
        ID, converting to .xyz and running crest in the scratch folder """

        self.run_hex = uuid.uuid4().hex[:6]
        self.xyz = tmpdirname + '/{0}_{1}.xyz'.format(self.cid, self.run_hex)
        self.crest_out = tmpdirname + '/{0}_{1}.crest.out'.format(self.cid, self.run_hex)
        self.best_xyz = tmpdirname + '/{0}_{1}_best.xyz.'.format(self.cid, self.run_hex)

        #writing to xyzfile
        rdkit.Chem.rdmolfiles.MolToXYZFile(mol,self.xyz,confId=confId)

        #running calc in temporary TemporaryDirectory : tmpdirname
        env = os.environ.copy()
        crest_cmd = "crest {0} > {1}".format(
            self.xyz, self.crest_out)

        subprocess.run(crest_cmd, shell=True, env=env,
                       timeout=self.crest_timeout)

        #crest outputs common name files such as crest_best.xyz. We have to move it
        #such that it can be accessed again to creat gjf file for gaussian
        subprocess.run(['mv', 'crest_best.xyz', self.best_xyz])



    def write_gaussian_input_file(self, tmpdirname):
        """ Given an best conformer after crest,
         write a gaussian input file using openbabel to the scratch folder """


        self.gjf = tmpdirname + '/{0}_{1}.gjf'.format(self.cid, self.run_hex)
        checkpoint_file = tmpdirname + '/{0}_{1}.chk'.format(self.cid, self.run_hex)

        if self.type_ == 'fragment':
            # Run stable=opt

            header1 = [
                '%chk={0}'.format(checkpoint_file),
                '%MEM={}'.format(self.mem),
                '%nprocshared={}'.format(self.nprocs),
                '# stable=opt M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400)'
                ' nosymm guess=mix']

            subprocess.run(
                ['obabel', '-ixyz',self.best_xyz, '-O', self.gjf, '-xk',
                 '\n'.join(header1)])

            with open(self.gjf, 'r') as f:
                chg_mul = f.readlines()[7]

            with open(self.gjf, 'a') as f:

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

            subprocess.run(
                ['obabel', '-ixyz',self.best_xyz, '-O', self.gjf, '-xk',
                 '\n'.join(header1)])


    def run_gaussian(self, tmpdirname):
        """ Run the given Guassian input file (with associated mol ID) """

        self.log = tmpdirname + '/{0}_{1}.log'.format(self.cid, self.run_hex)
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

        pt = Chem.GetPeriodicTable()
        covalent_radii = lambda x: PeriodicTable.GetRcovalent(pt, x)

        # Check bond lengths
        for bond in mol.GetBonds():
            length = GetBondLength(
                mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

            max_length = (covalent_radii(bond.GetBeginAtom().GetSymbol()) +
                          covalent_radii(bond.GetEndAtom().GetSymbol()) + 0.4)

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

        newlog = self.projectdir + 'log/' + log_basename + '.gz'
        newgjf = self.projectdir + 'gjf/' + gjf_basename + '.gz'

        subprocess.run(['gzip', self.log, self.gjf])
        subprocess.run(['mv', self.log + '.gz', newlog])
        subprocess.run(['mv', self.gjf + '.gz', newgjf])

        return newlog
