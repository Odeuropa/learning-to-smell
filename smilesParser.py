import sys, csv

class TBond:
    """
    'kind' = one of the strings '-' (single), '=' (double), '#' (triple), ':' (aromatic) etc.
    'node1', 'node2' = TNode objects connected by this bond
    """
    __slots__ = ["kind", "node1", "node2"]
    def __init__(self, kind, node1, node2):
        self.kind = kind; self.node1 = node1; self.node2 = node2
class TNode:
    """
    'atom' = a string containing the chemical symbol of this atom, e.g. "C", "N", "Br", etc.
    Aromatic atoms have a lowercase string: "c", "n".  A wildcard string "*" is also possible.
    'bonds' = a list of TBond objects having this node as one of the endpoints (node1 or node2)
    'nodeIdx' = index of this node in TGraph's 'nodes' list
    """
    __slots__ = ["atom", "bonds", "isAromatic", "nodeIdx"]
    def __init__(self, atom):
        self.atom = atom; self.bonds = []
        self.isAromatic = self.atom and (self.atom[0].islower() or self.atom[0] == '*')
        self.nodeIdx = -1 # this will be initialized by TGraph::AddNode
class TGraph:
    """
    'nodes' = a list of TNode objects
    'bonds' = a list of TBond objects
    'bondHash' = a dict with keys of the form (nodeIdx1, nodeIdx2) and with TBond objects as the corresponding values
    """
    __slots__ = ["nodes", "bonds", "bondHash"]
    def __init__(self): 
        self.nodes = []
        self.bonds = []
        self.bondHash = {} 
    def AddNode(self, node):
        node.nodeIdx = len(self.nodes)
        self.nodes.append(node); return node
    def GetNodeKey(self, node1, node2):
        idx1 = node1.nodeIdx; idx2 = node2.nodeIdx
        return min(idx1, idx2), max(idx1, idx2)
    def AddBond(self, kind, node1, node2):
        # Usually, if a bond between two atoms is not represented explicitly,
        # a single bond '-' should be assumed as the default.  However, if both
        # atoms were aromatic, then an aromatic bond ':' should be the default.
        if node1.isAromatic and node2.isAromatic and kind == '-': kind = ':'
        # Perhaps this bond already exists.
        key = self.GetNodeKey(node1, node2)
        bond = self.bondHash.get(key, None)
        if bond: assert bond.kind == kind; return bond
        bond = TBond(kind, node1, node2)
        self.bonds.append(bond); node1.bonds.append(bond); node2.bonds.append(bond)
        self.bondHash[key] = bond
        return bond
    def FindBond(self, node1, node2): 
        """returns a TBond object or none"""
        return self.bondHash.get(self.GetNodeKey(node1, node2), None)
    def Print(self, f = None):
        if not f: f = sys.stdout
        f.write("%d nodes, %d bonds\n" % (len(self.nodes), len(self.bonds)))
        for i, node in enumerate(self.nodes): 
            f.write("- Node %d:  %s\n" % (i, node.atom))
        for bond in self.bonds:
            f.write(" (%d) %s %s %s (%d)\n" % (bond.node1.nodeIdx, bond.node1.atom,
                bond.kind, bond.node2.atom, bond.node2.nodeIdx))

def ParseSmiles(s):
    """Parses the string 's' and returns a TGraph object.  The SMILES string format is described at:
    https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html """
    # 'i' indicates the current position in the string 's'.  The ParseXxxx local functions advance 'i'.
    s = s.strip(); n = len(s); i = 0; G = TGraph()
    nodesByTag = {}; PROVISIONAL_ATOM = "Provisional"
    validAtoms = set("* C N B F O P S Br Cl Zn c n o s")
    def ParseTag():
        """Parses a numeric tag of the form # or %##, where # is a digit, and returns the tag as a string.
        Returns an empty string if no tag was present at the current position."""
        nonlocal i
        if i >= n: return ""
        elif s[i] == '%':
            assert i + 2 <= n
            tag = int(s[i + 1:i + 3]); i += 3; return tag
        elif s[i].isdigit():
            tag = int(s[i]); i += 1; return tag
        else: return ""
    def ParseAtom():
        """Parses an atom, possibly in brackets, and returns its chemical symbol."""
        nonlocal i; assert i < n
        inBrackets = (s[i] == '[')
        if inBrackets: i += 1
        # Parse the element name.
        assert s[i].isalpha() or s[i] == '*'
        j = i + 1
        if s[i].isupper() and i + 1 < n and s[i:i + 2] in validAtoms: j = i + 2
        atom = s[i:j]; assert atom in validAtoms
        i = j
        # Skip the rest of the contents of the brackets (hydrogen atoms, charges).
        if inBrackets:
            while i < n and s[i] != ']': i += 1
            if i < n and s[i] == ']': i += 1
        return atom
    def ParseBond():
        """Parses a bond and returns it as a string.  Some normalization is also performed:
        '/' and '\' turn into '-'.  If there is no bond character at the current position, '-' is returned."""
        nonlocal i
        if i >= n: return "-"
        c = s[i]
        if c in ".=#$:": i += 1; return c
        if c in "-/\\": i += 1; return '-'
        if c == '@': # We'll ignore the chirality details for now.
            i += 1
            if i < n and s[i] == c: i += 1
            return '-'
        return '-'
    def ParseChain(prevNode):
        """Parses a chain (sequence) of atoms and connects them to 'prevNode'.  If 'prevNode' is None,
        the chain must not begin with a bond specification."""
        nonlocal i
        while i < n:
            if s[i] == ')': break
            # Parse the next bond and atom.
            bond = ParseBond()
            # Next we might have a tag that refers to some other atom; note that sometimes
            # this atom will only appear later in the formula, e.g.: O=CCC/C(=C/1\CCC(=CC1)C)/C .
            # In those cases we'll add a provisional atom.
            tag = ParseTag()
            if tag:
                if tag in nodesByTag:
                    assert prevNode
                    G.AddBond(bond, prevNode, nodesByTag[tag])
                    del nodesByTag[tag]
                    assert tag not in nodesByTag
                else:
                    nodesByTag[tag] = G.AddNode(TNode(PROVISIONAL_ATOM))
                continue
            # If not, there will be a new atom.
            atom = ParseAtom(); curNode = None
            # Read any tags that this atom may be associated with.
            tags = set()
            while True:
                tag = ParseTag()
                if not tag: break
                otherNode = nodesByTag.get(tag, None)
                # Perhaps one of the tags indicates that this is an atom that we have
                # already added provisionally at an earlier point.
                if otherNode and otherNode.atom == PROVISIONAL_ATOM:
                    #print("Changing provisional to normal for tag %d, thereby consuming it" % tag)
                    assert not curNode; curNode = otherNode
                    otherNode.atom = atom
                    del nodesByTag[tag]
                else: 
                    assert tag not in tags
                    tags.add(tag)
            # If this atom was not one of the previously provisional ones, add it now.
            if not curNode: 
                curNode = TNode(atom); G.AddNode(curNode)
            # Process the remaining tags.  They either label the current atom (if new)
            # or connect it to existing atoms with those tags.
            for tag in tags:
                if tag in nodesByTag:
                    G.AddBond('-', curNode, nodesByTag[tag])
                    del nodesByTag[tag]
                else:
                    nodesByTag[tag] = curNode
            # Connect the current atom to the previous one.
            if prevNode: G.AddBond(bond, prevNode, curNode)
            # Process subbranches (in parentheses) recursively.
            while i < n and s[i] == '(':
                i += 1; ParseChain(curNode)
                assert i < n and s[i] == ')'
                i += 1
            # Move to the next step of the chain.    
            prevNode = curNode
    ParseChain(None)
    # At this point, there should be no provisional atoms and 'nodesByTag' should be empty,
    # otherwise the string is invalid (e.g. an atom was tagged but no other atom used the tag to link to it).
    assert not nodesByTag
    for node in G.nodes: assert node.atom != PROVISIONAL_ATOM
    return G

def Test(fileName):
    with open(fileName, newline = '') as f:
        firstRow = True
        for row in csv.reader(f):
            # Skip the first row as it contains headers.
            if firstRow: firstRow = False; continue
            # The first entry in each row is a SMILES string.  The second entry, if present,
            # is a list of smells.
            smilesString = row[0]; smells = "" if len(row) <= 1 else row[1]
            print(smilesString)
            G = ParseSmiles(smilesString)

if __name__ == "__main__":
    # Assume that the .csv files from https://www.aicrowd.com/challenges/learning-to-smell/dataset_files
    # are available in the current working directory.
    Test("train.csv")
    Test("test.csv")
    for s in [
        #"C1CCC(CC1)O"
        #"O=C[C@@H]([C@H]([C@@H]([C@H](OC(=O)C)COC(=O)C)OC(=O)C)OC(=O)C)OC(=O)C",
        #"CC(C)c1cccc(c(=O)c1)O",
        #"OC[C@H]1[C@H]2CC[C@H]3[C@@]1(C)CCCC([C@@H]23)(C)C"
        "CC(C)Cc1ccnc2ccccc12"
        #"CC1CCc2c(C1)occ2C.CC1CCC(C(C1)OC(=O)C)C(C)C.CC1CCC(=C(C)C)C(=O)C1.CC1CCC(C(=O)C1)C(C)C.CC1CCC(C(C1)O)C(C)C.CC1CCC2(CC1)OCC2C"
        #"C[C@@H]1CC[C@@]23C[C@H]1C(C)(C)[C@@H]3CC/C/2=C/C(=O)C"
        #"CC(C)C1CC=C(C)c2ccc(C)cc12"
        #"O=CCC/C(=C/1\CCC(=CC1)C)/C"
        #"*OCOC"
        #" CC1C=CCCC1C1OCC(CO)O1"
        ]:
        G = ParseSmiles(s)
        print("\nSMILES string: %s" % s)
        G.Print()

