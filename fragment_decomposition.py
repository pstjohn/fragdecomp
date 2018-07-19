from collections import Counter
import re

import pandas as pd
import networkx as nx

from rdkit import Chem
# from IPython.display import SVG

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from fragdecomp.chemical_conversions import canonicalize_smiles

class FragmentError(Exception):
    pass


def get_fragments(smiles, size=1):
    """Return a pandas series indicating the carbon types in the given SMILES
    string

    smiles: string
        A representation of the desired molecule. I.e, 'CCCC'

    """
    try:
        test = Fragmenter(smiles, size=size)
        return test.fragment_counts

    
    except Exception:
        # Deal with wierd rdkit errors
        raise FragmentError


class Fragmenter(object):
    def __init__(self, smiles, size=1):
        
        self.size = size
        
        # Build rdkit mol
        mol = Chem.MolFromSmiles(canonicalize_smiles(smiles, isomeric=False))
        self.mol = mol = Chem.AddHs(mol)
        
        # Build nx graph
        self.G = nx.Graph()
        
        for atom in mol.GetAtoms():
            self.G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())

        for i, bond in enumerate(mol.GetBonds()):
            self.G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
        c_nodes = [n for n in self.G if self.G.nodes[n]['symbol'] is 'C']
        self.H = self.G.subgraph(c_nodes)

        
    def backbone_iterator(self, size, current_backbone=[]):

        if current_backbone == []:
            next_nodes = self.H
        else:
            next_nodes = self.H.neighbors(current_backbone[-1])

        for n in next_nodes:

            if n in current_backbone:
                continue

            if len(current_backbone + [n]) == size:
                yield tuple(sorted(current_backbone + [n]))

            else:
                for path in self.backbone_iterator(
                        size, current_backbone + [n]):
                    yield path

    def backbones(self, size):
        return list(set(self.backbone_iterator(size)))
    
    def smiles_from_backbone(self, backbone):

        fragment_atoms = set()
        for atom in backbone:
            fragment_atoms |= {atom}
            fragment_atoms |= set(self.G.neighbors(atom))
                
        smarts = Chem.MolFragmentToSmiles(
            self.mol, fragment_atoms, canonical=True,
            allBondsExplicit=True, allHsExplicit=True)
        
        ring = any([self.mol.GetAtomWithIdx(i).IsInRing() for i in backbone])
        if ring:
            smarts = smarts + ' | (Ring)'
            
        return smarts
    
    @property
    def fragment_counts(self):
        return pd.Series(Counter((
            self.smiles_from_backbone(item)
            for item in self.backbones(self.size))))


# def label_fragments(smiles):
#     """For a given smiles string, return the carbon fragments and atom indices
#     corresponding to those fragments
#
#     smiles: string
#
#     """
#     mol = Chem.MolFromSmiles(canonicalize_smiles(smiles, isomeric=False))
#     mol = Chem.AddHs(mol)
#     out = {}
#     for carbon in iter_carbons(mol):
#         try:
#             out[get_environment_smarts(carbon, mol)] += [carbon.GetIdx()]
#         except KeyError:
#             out[get_environment_smarts(carbon, mol)] = [carbon.GetIdx()]
#     return pd.Series(out)


def draw_mol_svg(mol_str, color_dict=None, figsize=(300, 300), smiles=True):
    """Return an SVG image of the molecule, with atoms highlighted to reveal
    the sootiest fragments.

    (Note: This is currently not very quantitative...)

    mol_str: string
    fragment_soot: dict-like
        A dictionary matching fragment SMILES to color
    figsize: tuple
        Figure size (in pixels) to pass to rdkit.

    """
    if smiles:
        mol = Chem.MolFromSmiles(mol_str)
    else:
        mol = Chem.MolFromSmarts(mol_str)

    mc = Chem.Mol(mol.ToBinary())
    if True:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())

    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)

    if color_dict is not None:
        matches = label_fragments(mol_str)
        highlights = flatten(matches.values.tolist())
        highlight_colors = [tuple(color_dict[matches.index[i]]) for i, match in
                            enumerate(matches.values.tolist()) for atom in match]
        highlight_colors_dict = {atom_id: color for atom_id, color
                                 in zip(highlights, highlight_colors)}
        
        drawer.DrawMolecule(
            mc, highlightAtoms=highlights,
            highlightAtomColors=highlight_colors_dict,
            highlightBonds=False)

    else:
        drawer.DrawMolecule(mc)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    svg = svg.replace('svg:', '').replace(':svg', '')
    return svg


def flatten(l, ltypes=(list, tuple)):
    """Utility function to iterate over a flattened list"""
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def draw_fragment(fragment_name, color):
    
    mol = Chem.MolFromSmarts(re.sub(' \|.*$', '', fragment_name))
    mc = Chem.Mol(mol.ToBinary())
    rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DSVG(80, 80)

    center = int(pd.Series({atom.GetIdx(): len(atom.GetNeighbors()) for atom in
                            mol.GetAtoms()}).idxmax())
    
    to_highlight = [center]
    radius_dict = {center: 0.5}
    color_dict = {center: color}
    
    drawer.DrawMolecule(mc, highlightAtoms=to_highlight,
                        highlightAtomColors=color_dict,
                        highlightAtomRadii=radius_dict,
                        highlightBonds=False)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg.replace('svg:', '').replace(':svg', '')

