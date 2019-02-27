import os
import tempfile
import subprocess
import logging
import uuid

import numpy as np
import cclib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetBondLength

logging.getLogger("cclib").setLevel(30)

def optimize_molecule_mmff(smiles, max_conformers=1000):
    """ Embed a molecule in 3D space, optimizing a number of conformers and
    selecting the most stable
    """
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.rdmolops.AddHs(mol)

    NumRotatableBonds = int(AllChem.CalcNumRotatableBonds(mol))
    NumConformers = min(3**NumRotatableBonds, max_conformers)

    conformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=NumConformers, pruneRmsThresh=0.2, randomSeed=1,
        useExpTorsionAnglePrefs=True, useBasicKnowledge=True)

    assert conformers, "Conformer embedding failed"

    def optimize_conformer(conformer):
        prop = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=conformer)
        ff.Minimize()
        return float(ff.CalcEnergy())

    conformer_energies = np.array(
        [optimize_conformer(conformer) for conformer in conformers])

    most_stable_conformer = conformer_energies.argmin()
    return mol, int(most_stable_conformer)


def write_gaussian_input_file(mol, confId, cid, scratchdir='/scratch/pstjohn',
                              nprocs=18, mem='24GB'):
    """ Given an rdkit.Mol object with an optimized, minimum energy conformer
    ID, write a gaussian input file using openbabel to the scratch folder """

    run_hex = uuid.uuid4().hex[:6]
    input_file = scratchdir + '/bde/gjf/{0}_{1}.gjf'.format(cid, run_hex)
    checkpoint_file = scratchdir + '/bde/chk/{0}_{1}.chk'.format(cid, run_hex)
    
    header1 = [
        '%chk={0}'.format(checkpoint_file),        
        '%MEM={}'.format(mem),
        '%nprocshared={}'.format(nprocs),
        '# stable=opt M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400)'
        ' nosymm guess=mix']
    
    with tempfile.NamedTemporaryFile(
            'w', suffix='.sdf', dir=scratchdir + '/gauss_scr') as sdf_file:
        writer = Chem.SDWriter(sdf_file)
        mol.SetProp('_Name', str(cid))
        writer.write(mol, confId=confId)
        writer.close()

        subprocess.call(
            ['obabel', sdf_file.name, '-O', input_file, '-xk', '\n'.join(header1)])

        
    with open(input_file, 'r') as f:
        chg_mul = f.readlines()[7]

    with open(input_file, 'a') as f:
        
        header2 = [
            '--link1--',
            '%chk={0}'.format(checkpoint_file),        
            '%MEM={}'.format(mem),
            '%nprocshared={}'.format(nprocs),
            '# opt freq M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400)'
                ' nosymm guess=read geom=check\n',
            ' {}\n'.format(mol.GetProp('_Name')),
            chg_mul
        ]
        
        f.write('\n'.join(header2))
        
    return input_file, run_hex


def run_gaussian(gjf, cid, run_hex, scratchdir='/scratch/pstjohn'):
    """ Run the given Guassian input file (with associated mol ID) """

    output_file = scratchdir + '/bde/log/{0}_{1}.log'.format(cid, run_hex)
    
    with tempfile.TemporaryDirectory(dir=scratchdir + '/gauss_scr') as tmpdirname:
        env = os.environ.copy()
        env['GAUSS_SCRDIR'] = tmpdirname
        subprocess.call("module load gaussian/G16B && g16 < {0} > {1}".format(
            gjf, output_file), shell=True, env=env)

    return output_file


def parse_log_file(logfile, smiles, cid):
    """ Parse the gaussian log file using cclib, return the optimized mol and
    enthalpy. """

    # Parse the output log with cclib, assert the optimization completed
    data = cclib.io.ccread(logfile)
    assert data.optdone, "Optimization not converged"
    
    if hasattr(data, 'vibfreqs'): # single atoms don't have this property
        assert min(data.vibfreqs) >= 0, "Imaginary Frequency"

    # Create an RDKit Molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
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
    mol.SetProp("_Name", str(cid))
    mol.SetProp('SMILES', smiles)
    mol.SetDoubleProp('Enthalpy', data.enthalpy)

    return mol, data.enthalpy
