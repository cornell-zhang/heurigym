from Bio.PDB import PDBList, PDBParser, Select, PDBIO
import os

# PDBs with monomer/dimer variants
monomer_dimer_chains = {
    "1arr": {"monomer": ["A"], "dimer": ["A", "B"]},
    "1cmb": {"monomer": ["A"], "dimer": ["A", "B"]},
    "1kba": {"monomer": ["A"], "dimer": ["A", "B"]},
}

# All PDB entries from your list
pdb_ids = [
    "1aaj", "1aba", "1aps", "1arr", "1bba", "1bbl", "1bov", "1brq",
    "1cis", "1cmb", "1hel", "1ifb", "1kba", "2gbl", "2hpr",
    "2il8", "2s6b", "3cln", "3rn3", "3trx"
]

os.makedirs("pdb_structures", exist_ok=True)

pdbl = PDBList()
for pdb_id in set(pdb_ids):
    try:
        pdbl.retrieve_pdb_file(pdb_id, pdir="pdb_structures", file_format="pdb")
    except Exception as e:
        print(f"⚠️ Failed to download {pdb_id}: {e}")

# PDB parser + writer
parser = PDBParser(QUIET=True)
io = PDBIO()

class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = chains
    def accept_chain(self, chain):
        return chain.id in self.chains

# Loop through and process existing files
for pdb_id in pdb_ids:
    path = f"pdb_structures/pdb{pdb_id}.ent"
    if not os.path.exists(path):
        print(f"❌ Skipping missing file for {pdb_id}")
        continue

    structure = parser.get_structure(pdb_id, path)

    if pdb_id in monomer_dimer_chains:
        for form, chains in monomer_dimer_chains[pdb_id].items():
            io.set_structure(structure)
            out_path = f"pdb_structures/{pdb_id}_{form}.pdb"
            io.save(out_path, ChainSelect(chains=chains))
    else:
        io.set_structure(structure)
        out_path = f"pdb_structures/{pdb_id}.pdb"
        io.save(out_path)

print("✅ Done processing available structures.")
