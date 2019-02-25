from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetBondLength
import cclib

import os
import tempfile
import subprocess
import logging

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

    def optimize_conformer(conformer):
        prop = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=conformer)
        ff.Minimize()
        return float(ff.CalcEnergy())

    conformer_energies = np.array(
        [optimize_conformer(conformer) for conformer in conformers])

    most_stable_conformer = conformer_energies.argmin()
    return mol, int(most_stable_conformer)


def write_gaussian_input_file(mol, confId, cid, scratchdir='/scratch/pstjohn'):

    input_file = scratchdir + '/bde/gjf/{}_0.gjf'.format(cid)
    checkpoint_file = scratchdir + '/bde/chk/{0}_0.chk'.format(cid)
    
    header1 = [
        '%chk={0}'.format(checkpoint_file),        
        '%MEM=24GB',
        '%nprocshared=18',
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
            '# opt freq M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400)'
                ' nosymm guess=read geom=check\n',
            ' {}\n'.format(mol.GetProp('_Name')),
            chg_mul
        ]
        
        f.write('\n'.join(header2))
        
    return input_file


def run_gaussian(gjf, cid, scratchdir='/scratch/pstjohn'):

    output_file = scratchdir + '/bde/log/{}_0.log'.format(cid)
    
    with tempfile.TemporaryDirectory(dir=scratchdir + '/gauss_scr') as tmpdirname:
        env = os.environ.copy()
        env['GAUSS_SCRDIR'] = tmpdirname
        subprocess.call("module load gaussian/G16B && g16 < {0} > {1}".format(
            gjf, output_file), shell=True, env=env)
    
    
