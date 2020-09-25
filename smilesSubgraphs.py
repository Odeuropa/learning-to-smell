import sys, csv, itertools, math, random
import smilesParser
import scipy, scipy.sparse
import sklearn, sklearn.svm, sklearn.calibration

def ExpandSubset(G, nodeSet):
    """This function yields all the sets that can be obtained by expanding
    'nodeSet' with a neighbour of one of its current members."""
    # To reduce the number of times we generate the same subset, we enforce
    # a rule that the member with the lowest nodeIdx must be visited first.
    minNodeIdx = min(nodeSet)
    newSet = set(nodeSet)
    for nodeIdx in nodeSet:
        for bond in G.nodes[nodeIdx].bonds:
            if bond.kind == '.': continue # not really a bond at all!
            otherNodeIdx = bond.node2.nodeIdx
            if otherNodeIdx <= minNodeIdx or otherNodeIdx in nodeSet: continue
            newSet.add(otherNodeIdx)
            yield frozenset(newSet)
            newSet.remove(otherNodeIdx)

def ExpandSubsets(G, subsets):
    """Given a set of frozensets of nodes, this function returns a set of frozensets
    obtained by expanding the input frozensets with one more node each."""
    newSubsets = set()
    for oldSubset in subsets:
        for newSubset in ExpandSubset(G, oldSubset):
            newSubsets.add(newSubset)
    return newSubsets        
    
def GenSubgraphs(G, maxSubgraphSize):
    """Returns a set of frozensets representing all the connected subgraphs of G
    from 1 to maxSubgraphSize nodes."""
    # Each node already has a 0-based nodeIdx.  
    nodeSubsets = set(frozenset([i]) for i in range(len(G.nodes)))
    allSubsets = nodeSubsets.copy()
    totalSubsets = len(nodeSubsets)
    for k in range(1, maxSubgraphSize):
        nodeSubsets = ExpandSubsets(G, nodeSubsets)
        #print("%d subsets of size %d." % (len(nodeSubsets), k + 1))
        allSubsets |= nodeSubsets
        totalSubsets += len(nodeSubsets)
    assert len(allSubsets) == totalSubsets    
    return allSubsets

def SubgraphToString(G, nodeSet):
    """This function returns a canonical string representing the subgraph that is induced by the set 'nodeSet' on G.
    The string is canonical in the sense that if you were to renumber the nodes, the resulting string would remain the same.
    We do this by first ordering the nodes by (atom, degree in G, degree in the subgraph) triples; 
    if several nodes have the same triple, we try ordering them in all possible permutations.
    For each resulting order, we generate a string listing the nodes and edges of the subgraph.
    Among the resulting strings, we return the lexicographically smallest one.
    In the worst case, we may have to examine O(k!) orders, where k = len(nodeSet).  Most of the time it's much less than that, however.
    """
    # Calculate a simple string describing each node.
    nodesWithLabels = {}
    bonds = []
    for nodeIdx in nodeSet:
        node = G.nodes[nodeIdx]
        nNeigh = 0; nNeighInSubgraph = 0
        for bond in node.bonds:
            if bond.kind == '.': continue
            nNeigh += 1; otherNodeIdx = bond.OtherNode(node).nodeIdx
            if otherNodeIdx not in nodeSet: continue
            nNeighInSubgraph += 1
            bonds.append((nodeIdx, otherNodeIdx, bond.kind))
        nodeLabel = "%s %d %d" % (node.atom, nNeigh, nNeighInSubgraph)
        if nodeLabel not in nodesWithLabels: nodesWithLabels[nodeLabel] = []
        nodesWithLabels[nodeLabel].append(nodeIdx)
    if False:
        n = 1
        for (label, L) in nodesWithLabels.items(): n *= math.factorial(len(L))
        if n >= 362880: print(nodesWithLabels)
        return n
    # We'll list the nodes in the order of their string labels.
    nodesWithLabels = list(nodesWithLabels.items()); nodesWithLabels.sort()
    labelString = ",".join(label for (label, L) in nodesWithLabels for nodeIdx in L)
    allOrders = [tuple()]
    for (label, L) in nodesWithLabels:
        allOrders = [order + Lorder for order in allOrders for Lorder in itertools.permutations(L)]
    bestBondString = ""    
    for order in allOrders:
        nodeIdxToOrderIdx = {}
        for i, nodeIdx in enumerate(order): nodeIdxToOrderIdx[nodeIdx] = i
        bonds2 = []
        for (nodeIdx1, nodeIdx2, kind) in bonds:
            orderIdx1 = nodeIdxToOrderIdx[nodeIdx1]
            orderIdx2 = nodeIdxToOrderIdx[nodeIdx2]
            bonds2.append("%s %d %d" % (kind, min(orderIdx1, orderIdx2), max(orderIdx1, orderIdx2)))
        bonds2.sort()
        bondString = ",".join(bonds2)
        if not bestBondString or bondString < bestBondString: bestBondString = bondString
    return "%s;%s" % (labelString, bestBondString)

class TDocSet:
    """Represents a set of documents.  For document i, its sparse vector consists of the 
    feature indices csrIndices[csrIndPtr[i]:csrIndPtr[i + 1]] and the corresponding
    feature values csrData[csrIndPtr[i]:csrIndPtr[i + 1]].  Furthermore, classLists[i]
    is the list of 0-based IDs of classes to which this document belongs, and smilesStrings[i]
    is the SMILES string representing this document.  Thus the TDocSet instance makes sense
    only in association with a TDataset instance, otherwise you can't convert feature IDs
    to subgraphs or class IDs to their names."""
    __slots__ = ["csrData", "csrIndices", "csrIndPtr", "classLists", "smilesStrings"]
    def __init__(self):
        self.csrData = []
        self.csrIndices = []
        self.csrIndPtr = [0]
        self.classLists = []
        self.smilesStrings = []
    def Add(self, tfVec, classList, smilesString):
        L = list(tfVec.items()); L.sort()
        for (featIdx, featVal) in L:
            self.csrData.append(featVal); self.csrIndices.append(featIdx)
        self.csrIndPtr.append(len(self.csrData))
        self.classLists.append(classList[:])
        self.smilesStrings.append(smilesString)
    def GetCsrMatrix(self, nFeatures):
        print("GetCsrMatrix(%d docs)." % len(self.classLists))
        return scipy.sparse.csr_matrix((self.csrData, self.csrIndices, self.csrIndPtr), shape=(len(self.smilesStrings), nFeatures))
    def GetClassColumns(self, nClasses):
        print("GetClassColumns(%d docs, %d classes)." % (len(self.classLists), nClasses))
        nDocs = len(self.classLists)
        L = [[0] * nDocs for i in range(nClasses)]
        for docNo, classList in enumerate(self.classLists):
            for classNo in classList: L[classNo][docNo] = 1
        return L
    def StratifiedSplit(self, classNames):
        """Returns two sets of document numbers representing a somewhat balanced split of this document set.
        This method starts with two empty subsets, then processes the classes from smallest to largest.  For
        each class, it randomly assigns document from that class to the first subset until half of the class
        is in that subset.  At the end of this process, anything that hasn't been assigned into the first 
        subset is considered to form the second subset.  Thus the first subset is likely to contain a bit
        more than half of all documents."""
        r = random.Random(123)
        nDocs = len(self.classLists)
        classFreqs = []; docsByClass = []
        for docNo, classList in enumerate(self.classLists):
            for classNo in classList:
                while len(classFreqs) <= classNo: classFreqs.append(0); docsByClass.append([])
                classFreqs[classNo] += 1; docsByClass[classNo].append(docNo)
        nClasses = len(classFreqs)
        classFreqPrs = [(classFreqs[classNo], classNo) for classNo in range(nClasses)]
        classFreqPrs.sort(); classFreqPrs.reverse()
        classFreqs1 = [0] * nClasses; docAssignments = [0] * nDocs
        for (classFreq, classNo) in classFreqPrs:
            unassignedDocs = [docNo for docNo in docsByClass[classNo] if docAssignments[docNo] == 0]
            unassignedDocs.sort(); r.shuffle(unassignedDocs)
            for docNo in unassignedDocs:
                if classFreqs1[classNo] >= classFreq // 2: break
                docAssignments[docNo] = 1
                for classNo2 in self.classLists[docNo]: classFreqs1[classNo2] += 1
        if False:
            def _(s, a, b): print("[%s] %d -> %d + %d  (%.2f : %.2f)" % (s, a, b, a - b, 100.0 * b / float(a), 100.0 * (a - b) / float(a)))
            _("Total", nDocs, sum(docAssignments))
            for classNo in range(nClasses): _(classNames[classNo], classFreqs[classNo], classFreqs1[classNo])
        if False:
            ds1 = TDocSet(); ds2 = TDocSet()
            for docNo in len(nDocs):
                dest = ds1 if docAssignments[docNo] else ds2
                i1 = self.csrIndices[docNo]; i2 = self.csrIndices[docNo + 1]
                dest.csrData.append(self.csrData[i1 : i2])
                dest.csrIndices.append(self.csrIndices[i2 : i2])
                dest.csrIndPtr.append(len(dest.csrData))
                dest.classLists.append(self.classLists[docNo])
                dest.smilesStrings.append(self.smilesStrings[docNo])
            return (ds1, ds2)
        return set(docNo for docNo in range(nDocs) if docAssignments[docNo]), set(docNo for docNo in range(nDocs) if not docAssignments[docNo])

class TDataset:
    __slots__ = ["strToIdHash", "dfList", "nTrainDocs", "classToIdHash", "classNames"]
    def __init__(self):
        self.strToIdHash = {}
        self.dfList = []
        self.nTrainDocs = 0
        self.classToIdHash = {}
        self.classNames = []
    def GraphToTfVector(self, G, maxSubsetSize, allowNewFeatures, dfDelta):
        """Returns the TF-vector as a hash table of (featureId: TF) pairs."""
        self.nTrainDocs += dfDelta
        V = {}
        for nodeSet in GenSubgraphs(G, maxSubsetSize):
            featureStr = SubgraphToString(G, nodeSet)
            featureId = self.strToIdHash.get(featureStr, -1)
            if featureId < 0:
                if not allowNewFeatures: continue
                featureId = len(self.dfList)
                self.strToIdHash[featureStr] = featureId
                self.dfList.append(0)
            V[featureId] = V.get(featureId, 0) + 1
        for featureId in V: self.dfList[featureId] += dfDelta    
        return V
    def FileToDocSet(self, csvFileName, maxSubsetSize, allowNewFeatures, dfDelta, docsToUse = None):
        """Reads a CSV file and returns a new TDocSet instance representing its documents, with TF vectors.
        The caller is responsible for eventually calling ApplyIdf before preparing a document-term matrix.
        If allowNewFeatures is True, new features (representing subgraphs that are not yet in self.strToIdHash)
        will be added to self.strToIdHash, and likewise new classes will be added to self.classNames;
        otherwise such new features and classes will be simply ignored.
        'dfDelta' is the value to be added to the DF of a feature (in self.dfList) whenever this feature
        appears in a document.  Thus you should use allowNewFeatures = True, dfDelta = 1 for the training
        set and allowNewFeatures = False, dfDelta = 0 for the test set."""
        print("Reading \"%s\", maxSubsetSize = %d." % (csvFileName, maxSubsetSize))
        ds = TDocSet(); nDocsRead = 0
        for smilesString, smells in smilesParser.ReadCsv(csvFileName):
            if docsToUse is not None and nDocsRead not in docsToUse: nDocsRead += 1; continue
            smells = smells.strip().split(',')
            classList = []
            for smell in smells:
                smellId = self.classToIdHash.get(smell, -1)
                if smellId < 0:
                    if not allowNewFeatures: continue
                    smellId = len(self.classNames); self.classToIdHash[smell] = smellId; self.classNames.append(smell)
                classList.append(smellId)
            G = smilesParser.ParseSmiles(smilesString)
            tfVec = self.GraphToTfVector(G, maxSubsetSize, allowNewFeatures, dfDelta)
            ds.Add(tfVec, classList, smilesString)
            nDocsRead += 1
            #if nDocsRead >= 100: break
        print("%d documents read; %d features, %d classes are now known." % (len(ds.classLists), len(self.dfList), len(self.classNames)))
        return ds
    def ApplyIdf(self, docSet):
        print("Applying IDF to a set of %d documents read." % len(docSet.classLists))
        lnN = math.log(max(self.nTrainDocs, 1))
        idfList = [lnN - math.log(max(df, 1)) for df in self.dfList]
        data = docSet.csrData; indices = docSet.csrIndices
        for docNo in range(len(docSet.csrIndPtr) - 1):
            iFrom = docSet.csrIndPtr[docNo]; iTo = docSet.csrIndPtr[docNo + 1]
            if iTo <= iFrom: continue
            # Convert TF values to TF*IDV values.
            sumSq = 0
            for i in range(iFrom, iTo):
                x = data[i] * idfList[indices[i]]
                sumSq += x * x; data[i] = x
            # Normalize the vector.
            if sumSq > 0:
                coef = 1.0 / math.sqrt(sumSq)
                for i in range(iFrom, iTo): data[i] *= coef
    def WritePredictions(self, fileName, docSet, predsByClass):
        """Writes predictions to 'fileName' in the CSV format.  SMILES strings are taken from 'docSet'.
        For each document, the top 3 classes are output as a prediction.
        predsByClass[i][j] must be a real-valued prediction for class i, document j."""
        f = open(fileName, "wt"); f.write("SMILES,PREDICTIONS\n")
        nClasses = len(self.classNames); nDocs = len(docSet.smilesStrings)
        assert len(predsByClass) == nClasses
        for x in predsByClass: assert len(x) == nDocs
        avgJaccard = 0
        for docNo in range(nDocs):
            L = [(predsByClass[i][docNo], i) for i in range(nClasses)]
            L.sort(); L.reverse()
            predictions = [i for (dummy, i) in L[:3]]
            f.write("%s,\"%s\"\n" % (docSet.smilesStrings[docNo], ",".join(self.classNames[i] for i in predictions)))
            trueValues = docSet.classLists[docNo][:3]
            nIntersection = 0
            for prediction in predictions: 
                if prediction in trueValues: nIntersection += 1
            nUnion = len(predictions) + len(trueValues) - nIntersection
            jaccard = nIntersection / float(max(1, nUnion))
            avgJaccard += jaccard
        f.close()
        avgJaccard /= float(nDocs)
        print("%s: avg jaccard = %.4f" % (fileName, avgJaccard))

class TContTable:
    __slots__ = ["tp", "tn", "fp", "fn", "precision", "recall", "accuracy", "f1", "bep", "auroc", "bestF1", "bestF1thr", "nMac", "docs"]
    def __init__(self): 
        self.tp = 0; self.tn = 0; self.fp = 0; self.fn = 0
        self.precision = 0; self.recall = 0; self.accuracy = 0; self.f1 = 0; self.bep = 0; self.auroc = 0; self.bestF1 = 0
        self.nMac = 0; self.docs = []
    def AddDoc(self, isPositive, prediction):
        self.docs.append((prediction, isPositive))
        if prediction > 0:
            if isPositive: self.tp += 1
            else: self.fp += 1
        else:
            if isPositive: self.fn += 1
            else: self.tn += 1
    def AddDocs(self, y, z):
        nDocs = len(y); assert len(z) == nDocs
        for i in range(nDocs): self.AddDoc(y[i], z[i])
    def AddMic(self, ct): # adds the per-class contingency table 'ct' to 'self' with a view to computing microaverages
        self.tp += ct.tp; self.tn += ct.tn; self.fp += ct.fp; self.fn += ct.fn
    def AddMac(self, ct): # adds the per-class contingency table 'ct' to 'self' with a view to computing macroaverages
        self.nMac += 1; ct.Calc()
        self.precision += ct.precision; self.recall += ct.recall
        self.accuracy += ct.accuracy; self.f1 += ct.f1
        self.bep += ct.bep; self.bestF1 += ct.bestF1; self.auroc += ct.auroc
    def Calc(self):
        if self.nMac > 0:
            self.precision /= float(self.nMac); self.recall /= float(self.nMac)
            self.f1 /= float(self.nMac); self.accuracy /= float(self.nMac)
            self.bep /= float(self.nMac); self.auroc /= float(self.nMac); self.bestF1 /= float(self.nMac)
            return
        self.precision = 1 if self.tp + self.fp == 0 else self.tp / float(self.tp + self.fp)
        self.recall = self.tp / float(max(self.tp + self.fn, 1))
        self.accuracy = (self.tp + self.tn) / float(max(self.tp + self.fp + self.tn + self.fn, 1))
        self.f1 = (self.tp + self.tp) / float(max(self.tp + self.tp + self.fp + self.fn, 1))
        self.docs.sort(); tp = 0; fp = 0; tn = self.tn + self.fp; fn = self.tp + self.fn
        prec = 1; rec = 0; self.bestF1 = (tp + tp) / float(max(tp + tp + fp + fn, 1)); self.bestF1thr = 0 if not self.docs else self.docs[0][0] + 1
        bestGap = abs(prec - rec); self.bep = (prec + rec) * 0.5; self.auroc = 0
        self.docs.sort(); self.docs.reverse()
        for (prediction, isPositive) in self.docs:
            if isPositive: tp += 1; fn -= 1
            else: fp += 1; tn -= 1
            prec = tp / float(tp + fp); rec = tp / float(max(tp + fn, 1)); f1 = (tp + tp) / float(max(tp + tp + fp + fn, 1))
            if f1 > self.bestF1: self.bestF1 = f1; self.bestF1thr = prediction
            gap = abs(prec - rec)
            if gap < bestGap: bestGap = gap; self.bep = (prec + rec) * 0.5
            if isPositive: self.auroc += tn
        self.auroc /= float(max(1, (tp + fn) * (fp + tn)))
    def PrintStats(self, className):
        print("[%s] %d docs in class; tp = %d, fp = %d, tn = %d, fn = %d;  precision = %.04f, recall = %.04f, F1 = %.04f, accuracy = %.04f; bep = %.04f, bestF1 = %.04f, auroc = %.04f" % (
            className, self.tp + self.fn, self.tp, self.fp, self.tn, self.fn, self.precision, self.recall, self.f1, self.accuracy, self.bep, self.bestF1, self.auroc))

def Evaluate(className, y, z): # y = true class memberships {0, 1}; z = predictions (>0 or <0) 
    ct = TContTable()
    ct.AddDocs(y, z)
    ct.Calc()
    ct.PrintStats(className)
    return ct

def RunExperiment(maxSubsetSize = 7):
    if True: 
        # Split the training set into two halves and use one half for training and the other half for testing.
        # This could be useful for evaluation and parameter tuning.
        tempDataset = TDataset()
        fullTrainSet = tempDataset.FileToDocSet("train.csv", 1, True, 1)
        split1, split2 = fullTrainSet.StratifiedSplit(tempDataset.classNames)
        print("Split %d -> %d + %d docs." % (len(fullTrainSet.smilesStrings), len(split1), len(split2)))
        dataset = TDataset()
        trainSet = dataset.FileToDocSet("train.csv", maxSubsetSize, True, 1, split1)
        testSet = dataset.FileToDocSet("train.csv", maxSubsetSize, False, 0, split2)
    else:
        # Use the whole training set for training and compute predictions on the test set.
        dataset = TDataset()
        trainSet = dataset.FileToDocSet("train.csv", maxSubsetSize, True, 1)
        testSet = dataset.FileToDocSet("test.csv", maxSubsetSize, False, 0)
    # Prepare matrices of normalized TF-IDF vectors.
    dataset.ApplyIdf(trainSet)
    dataset.ApplyIdf(testSet)
    nClasses = len(dataset.classNames); nFeatures = len(dataset.dfList)
    trainX = trainSet.GetCsrMatrix(nFeatures); trainY = trainSet.GetClassColumns(nClasses)
    testX = testSet.GetCsrMatrix(nFeatures); testY = testSet.GetClassColumns(nClasses)
    # "F" in the variable names is for models obtained by fitting a sigmoid over the outputs of a linear SVM.
    trainZ = []; testZ = []; trainZF = []; testZF = []
    ctTrainMic = TContTable(); ctTrainMac = TContTable(); ctTrainMicF = TContTable(); ctTrainMacF = TContTable()
    ctTestMic = TContTable(); ctTestMac = TContTable(); ctTestMicF = TContTable(); ctTestMacF = TContTable()
    #
    for classNo, className in enumerate(dataset.classNames):
        y = trainY[classNo]; nTrainDocs = len(y); nPos = sum(y)
        # Train a plain SVM model.
        # Some preliminary experiments suggested that weighting the positive class with nTrainDocs/nPosDocs is too much,
        # but not weighting it at all is not ideal either.  sqrt(nTrainDocs/nPosDocs) looks OK.
        model = sklearn.svm.LinearSVC(C = 10, class_weight = {0: 1, 1: math.sqrt(nTrainDocs / float(max(nPos, 1)))})
        model.fit(trainX, y)
        # Optionally move the decision threshold to the point that maximizes F1 on the training set.
        # Experiments suggests that this just makes overfitting a bit worse.
        if False:
            z = model.decision_function(trainX)
            ct = Evaluate("Train  " + className, trainY[classNo], z)
            thresh = ct.bestF1thr
        else: thresh = 0
        # Calculate its predictions on the training set.
        z = model.decision_function(trainX)
        for i in range(len(z)): z[i] -= thresh
        trainZ.append(z)
        ct = Evaluate("Train  " + className, trainY[classNo], z)
        ctTrainMic.AddMic(ct); ctTrainMac.AddMac(ct)
        # Calculate its predictions on the test set.
        z = model.decision_function(testX)
        for i in range(len(z)): z[i] -= thresh
        testZ.append(z)
        ct = Evaluate("Test   " + className, testY[classNo], z)
        ctTestMic.AddMic(ct); ctTestMac.AddMac(ct)
        # Train a calibrated model that outputs probabilities.
        model = sklearn.svm.LinearSVC(C = 10, class_weight = {0: 1, 1: math.sqrt(nTrainDocs / float(max(nPos, 1)))})
        calib = sklearn.calibration.CalibratedClassifierCV(model, method = "sigmoid", cv = sklearn.model_selection.StratifiedKFold(n_splits = 3))
        calib.fit(trainX, y)
        # Calculate its predictions on the training set.  When computing F1 and the like, we'll take P > 0.5 as a positive prediction.
        thresh = 0.5
        z = calib.predict_proba(trainX); z = z[:, 1]; trainZF.append(z)
        for i in range(len(z)): z[i] -= thresh
        ct = Evaluate("TrainF " + className, trainY[classNo], z)
        ctTrainMicF.AddMic(ct); ctTrainMacF.AddMac(ct)
        # Calculate its predictions on the test set.
        z = calib.predict_proba(testX); z = z[:, 1]; testZF.append(z)
        for i in range(len(z)): z[i] -= thresh
        ct = Evaluate("TestF  " + className, testY[classNo], z)
        ctTestMicF.AddMic(ct); ctTestMacF.AddMac(ct)
    # Print micro- and macro-averages.
    ctTrainMic.Calc(); ctTrainMic.PrintStats("Train  Microaverages")
    ctTestMic.Calc(); ctTestMic.PrintStats("Test   Microaverages")
    ctTrainMac.Calc(); ctTrainMac.PrintStats("Train  Macroaverages")
    ctTestMac.Calc(); ctTestMac.PrintStats("Test   Macroaverages")
    ctTrainMicF.Calc(); ctTrainMicF.PrintStats("TrainF Microaverages")
    ctTestMicF.Calc(); ctTestMicF.PrintStats("TestF  Microaverages")
    ctTrainMacF.Calc(); ctTrainMacF.PrintStats("TrainF Macroaverages")
    ctTestMacF.Calc(); ctTestMacF.PrintStats("TestF  Macroaverages")
    # Output predictions and also print Jaccard scores if the true class membership is known.
    dataset.WritePredictions("predictions-train.csv", trainSet, trainZ)
    dataset.WritePredictions("predictions-test.csv", testSet, testZ)
    dataset.WritePredictions("predictions-trainF.csv", trainSet, trainZF)
    dataset.WritePredictions("predictions-testF.csv", testSet, testZF)

if __name__ == "__main__":
    RunExperiment(7)
