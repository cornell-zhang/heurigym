"""Help on module pytoulbar2:

NAME
    pytoulbar2 - Python3 interface of toulbar2.

DESCRIPTION


"""
from math import isinf
import traceback
try :
    import pytoulbar2.pytb2 as tb2
    tb2.init()
except Exception:
    traceback.print_exc()
    pass

class CFN:
    """pytoulbar2 base class used to manipulate and solve a cost function network.
    
    Constructor Args:
        ubinit (decimal cost or None): initial upper bound.
        resolution (int): decimal precision of costs.
        vac (int): if non zero, maximum solver depth minus one where virtual arc consistency algorithm is applied (1: VAC only in preprocessing).
        configuration (bool): if True then special settings for preference learning using incremental solving (see car configuration tutorial).
        vns (int or None): if None then solves using branch-and-bound methods else using variable neighborhood search heuristic
                           (-1: initial solution at random, -2: minimum domain values, -3: maximum domain values, 
                            -4: first solution found by DFS, >=0: or by LDS with at most vns discrepancies).
        seed (int): random seed.
        verbose (int): verbosity control (-1: no message, 0: search statistics, 1: search tree, 2-7: propagation information).
        init (bool): starts from scratch, forgets previous CFNs (default value is True). Must be set to False if this CFN depends on a previous CFN (e.g. this CFN is given as input to a AddWeightedCSPConstraint). 
    
    Members:
        CFN (WeightedCSPSolver): python interface to C++ class WeightedCSPSolver.
        
        Contradiction (exception): python exception corresponding to the same C++ class.
        
        Limit (exception|None): contains the last SolverOut exception or None if no exception occurs when solving with SolveNext.
        
        Option (TouBar2): python interface to C++ class ToulBar2.
        
        SolverOut (exception): python exception corresponding to the same C++ class.
        
        Top (decimal cost): maximum decimal cost (it can be used to represent a forbidden cost).
        
        Variables (dict): associative array returning the original domain (list or iterable) associated to a given variable name (str).
        
        VariableIndices (dict): associative array returning the variable name (str) associated to a given index (int).
        
        VariableNames (list): array of created variable names (str) sorted by their index number.
        
    See pytoulbar2test.py example in src repository.
    
    """
    def __init__(self, ubinit = None, resolution = 0, vac = 0, configuration = False, vns = None, seed = 1, verbose = -1, init=True):
        if init:
            tb2.reinit();
        tb2.option.decimalPoint = resolution   # decimal precision of costs
        tb2.option.vac = vac   # if no zero, maximum search depth-1 where VAC algorithm is performed (use 1 for preprocessing only)
        tb2.option.seed = seed    # random seed number (use -1 if a pseudo-randomly generated seed is wanted)
        tb2.option.verbose = verbose   # verbosity level of toulbar2 (-1:no message, 0:search statistics, 1:search tree, 2-7: propagation information)

        # default options (can be modified later by the user)
        tb2.option.FullEAC = False   # if True, exploit VAC integrality variable orderding heuristic or just Full-EAC heuristic if VAC diseable
        tb2.option.VACthreshold = False  # if True, reuse VAC auto-threshold value found in preprocessing during search 
        tb2.option.useRASPS = 0   # if 1 or greater, perform iterative RASPS depth-first search (or LDS if greater than 1) in preprocessing during 1000 backtracks to find a good initial upperbound (to be used with VAC)
        tb2.option.weightedTightness = 0   # if 1 or 2, variable ordering heuristic exploiting cost distribution information (0: none, 1: mean cost, 2: median cost)

        self.configuration = configuration   # if True then special settings for learning
        if configuration:
            tb2.option.elimDegree_preprocessing = 1   # maximum degree level of variable elimination in preprocessing (-1: none, 0: null degree, 1: degree one, etc.)
            tb2.option.solutionBasedPhaseSaving = False   #  if False do not reuse previous complete solutions as hints during incremental solving used by structure learning evaluation procedure!

        if vns is not None:
            tb2.option.vnsInitSol = vns   # if vns different than None then perform Decomposition-Guided Variable Neighborhood Search (-1: initial solution at random, -2: minimum domain values, -3: maximum domain values, -4: first solution found by DFS, >=0: or by LDS with at most vns discrepancies)
            tb2.option.lds = 4
            tb2.option.restart = 10000;
            tb2.option.searchMethod = 2;    # 0:DFBB or HBFS, 1:VNS, 2:DGVNS 4:Parallel DGVNS
            tb2.option.vnsNeighborVarHeur = 3;   # 0: random, 1:conflict, 3: decomposition

        self.Variables = {}
        self.VariableIndices = {}
        self.VariableNames = []
        
        self.CFN = tb2.Solver() # initialize VAC algorithm depending on tb2.option.vac
        self.InternalCFNs = list() # keep alive internal CFNs created by AddWeightedCSPConstraint
        
        self.UbInit = ubinit # warning! cannot convert initial upper bound into an integer cost before knowing the rest of the problem        
        self.Contradiction = tb2.Contradiction
        self.SolverOut = tb2.SolverOut
        self.Option = tb2.option
        self.Top = float('inf') # can be used to represent forbidden assignments
        self.Limit = None
        tb2.check()    # checks compatibility between selected options

    @staticmethod
    def flatten(S):
        """Warning: this recursive method might exceed maximum recursion depth
        """
        if S == []:
            return S
        if isinstance(S[0], list):
            return CFN.flatten(S[0]) + CFN.flatten(S[1:])
        return S[:1] + CFN.flatten(S[1:])

    def AddVariable(self, name, values):
        """AddVariable creates a new discrete variable.

        Args:
            name (str): variable name.
            values (list or iterable): list of domain values represented by numerical (int) or symbolic (str) values.

        Returns:
            Index of the created variable in the problem (int).
            
        Note:
            Symbolic values are implicitely associated to integer values (starting from zero) in the other functions.
            In case of numerical values, the initial domain size is equal to max(values)-min(values)+1 and not equal to len(values).
            Otherwise (symbolic case), the initial domain size is equal to len(values).

        """
        if name in self.Variables:
            raise RuntimeError(name+" already defined")
        self.Variables[name] = values

        if all(isinstance(value, int) for value in values):
            vIdx = self.CFN.wcsp.makeEnumeratedVariable(name, min(values), max(values))
            for vn in range(min(values), max(values)+1):
                self.CFN.wcsp.addValueName(vIdx, 'v' + str(vn))
                if vn not in values:
                    self.CFN.wcsp.remove(vIdx, vn)
        elif all(isinstance(value, str) for value in values):
            vIdx = self.CFN.wcsp.makeEnumeratedVariable(name, 0, len(values)-1)
            for vn in values:
                self.CFN.wcsp.addValueName(vIdx, vn)
        else:
                raise RuntimeError("Incorrect domain:"+str(values))
        self.VariableIndices[name] = vIdx
        self.VariableNames.append(name)
        return vIdx

    def AddFunction(self, scope, costs, incremental = False):
        """AddFunction creates a cost function in extension. The scope corresponds to the input variables of the function. 
        The costs are given by a flat array the size of which corresponds to the product of initial domain sizes (see note in AddVariable and also GetDomainInitSize). 
     
        Args:
            scope (list): input variables of the function. A variable can be represented by its name (str) or its index (int).
            costs (list): array of decimal costs for all possible assignments (iterating first over the domain values of the last variable in the scope).
            incremental (bool): if True then the function is backtrackable (i.e., it disappears when restoring at a lower depth, see Store/Restore).  

        Example:
            AddFunction(['x','y'], [0,1,1,0]) encodes a binary cost function on Boolean variables x and y such that (x=0,y=0) has a cost of 0,
            (x=0,y=1) has a cost of 1, (x=1,y=0) has a cost of 1, and (x=1,y=1) has a cost of 0.
           
        """
        sscope = set(scope)
        if len(scope) != len(sscope):
            raise RuntimeError("Duplicate variable in scope:"+str(scope))
        iscope = []
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices.get(v, -1)
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v) 
            
        if (len(iscope) == 0 and isinstance(costs, (int, float))):
            self.CFN.wcsp.postNullaryConstraint(costs)
            return
        mincost = min(costs)
        maxcost = max(costs)
        if (mincost == maxcost):
            self.CFN.wcsp.postNullaryConstraint(mincost)
            return
        assert(len(iscope) >= 1)
        if (len(iscope) == 1):
            assert(self.CFN.wcsp.getDomainInitSize(iscope[0]) == len(costs))
            self.CFN.wcsp.postUnaryConstraint(iscope[0], costs, incremental)
        elif (len(iscope) == 2):
            assert(self.CFN.wcsp.getDomainInitSize(iscope[0]) * self.CFN.wcsp.getDomainInitSize(iscope[1]) == len(costs))
            self.CFN.wcsp.postBinaryConstraint(iscope[0], iscope[1], costs, incremental)
        elif (len(iscope) == 3):
            assert(self.CFN.wcsp.getDomainInitSize(iscope[0]) * self.CFN.wcsp.getDomainInitSize(iscope[1]) * self.CFN.wcsp.getDomainInitSize(iscope[2]) == len(costs))
            self.CFN.wcsp.postTernaryConstraint(iscope[0], iscope[1], iscope[2], costs, incremental)
        else:
            if incremental:
                raise NameError('Sorry, incremental ' + str(len(iscope)) + '-arity cost functions not implemented yet in toulbar2.')
            self.CFN.wcsp.postNullaryConstraint(mincost)
            idx = self.CFN.wcsp.postNaryConstraintBegin(iscope, 0, len(costs) - costs.count(0))
            tuple = [self.CFN.wcsp.toValue(v, 0) for v in iscope]
            for cost in costs:
                if (isinf(cost)):
                    self.CFN.wcsp.postNaryConstraintTuple(idx, tuple, tb2.MAX_COST)
                elif cost > mincost:
                    self.CFN.wcsp.postNaryConstraintTuple(idx, tuple, int((cost-mincost) * 10 ** tb2.option.decimalPoint))
                for r in range(len(iscope)):
                    i = len(iscope)-1-r
                    v = iscope[i]
                    if tuple[i] < self.CFN.wcsp.toValue(v, self.CFN.wcsp.getDomainInitSize(v) - 1):
                        tuple[i] += 1
                        for j in range(i+1,len(iscope)):
                            tuple[j] = self.CFN.wcsp.toValue(iscope[j], 0)
                        break
            self.CFN.wcsp.postNaryConstraintEnd(idx)

    def AddCompactFunction(self, scope, defcost, tuples, tcosts, incremental = False):
        """AddCompactFunction creates a cost function in extension. The scope corresponds to the input variables of the function. 
        The costs are given by a list of assignments with the corresponding list of costs, all the other assignments taking the default cost. 
     
        Args:
            scope (list): input variables of the function. A variable can be represented by its name (str) or its index (int).
            defcost (decimal cost): default cost.
            tuples (list): array of assignments (each assignment is a list of domain values, following the scope order).
            tcosts (list): array of corresponding decimal costs (tcosts and tuples have the same size).
            incremental (bool): if True then the function is backtrackable (i.e., it disappears when restoring at a lower depth, see Store/Restore).  

        Example:
            AddCompactFunction(['x','y','z'],0,[[0,0,0],[1,1,1]],[1,-1]) encodes a ternary cost function with the null assignment having a cost of 1,
            the identity assignment having a cost of -1, and all the other assignments a cost of 0.
           
        """
        assert(len(tuples) == len(tcosts))
        sscope = set(scope)
        if len(scope) != len(sscope):
            raise RuntimeError("Duplicate variable in scope:"+str(scope))
        iscope = []
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices.get(v, -1)
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v) 
            
        if (len(iscope) == 0):
            assert(len(tuples) == 0)
            self.CFN.wcsp.postNullaryConstraint(defcost)
            return
        mincost = min(defcost, min(tcosts))
        maxcost = max(defcost, max(tcosts))
        if (mincost == maxcost):
            self.CFN.wcsp.postNullaryConstraint(mincost)
            return
        assert(len(iscope) >= 1)
        if (len(iscope) == 1):
            costs = [defcost] * self.CFN.wcsp.getDomainInitSize(iscope[0])
            for i, tuple in enumerate(tuples):
                costs[self.CFN.wcsp.toIndex(iscope[0], tuple[0])] = tcosts[i]
            self.CFN.wcsp.postUnaryConstraint(iscope[0], costs, incremental)
        elif (len(iscope) == 2):
            costs = [defcost] * (self.CFN.wcsp.getDomainInitSize(iscope[0]) * self.CFN.wcsp.getDomainInitSize(iscope[1]))
            for i, tuple in enumerate(tuples):
                costs[self.CFN.wcsp.toIndex(iscope[0], tuple[0]) * self.CFN.wcsp.getDomainInitSize(iscope[1]) + self.CFN.wcsp.toIndex(iscope[1], tuple[1])] = tcosts[i]
            self.CFN.wcsp.postBinaryConstraint(iscope[0], iscope[1], costs, incremental)
        elif (len(iscope) == 3):
            costs = [defcost] * (self.CFN.wcsp.getDomainInitSize(iscope[0]) * self.CFN.wcsp.getDomainInitSize(iscope[1]) * self.CFN.wcsp.getDomainInitSize(iscope[2]))
            for i, tuple in enumerate(tuples):
                costs[self.CFN.wcsp.toIndex(iscope[0], tuple[0]) * self.CFN.wcsp.getDomainInitSize(iscope[1]) * self.CFN.wcsp.getDomainInitSize(iscope[2]) + self.CFN.wcsp.toIndex(iscope[1], tuple[1]) * self.CFN.wcsp.getDomainInitSize(iscope[2]) + self.CFN.wcsp.toIndex(iscope[2], tuple[2])] = tcosts[i]
            self.CFN.wcsp.postTernaryConstraint(iscope[0], iscope[1], iscope[2], costs, incremental)
        else:
            if incremental:
                raise NameError('Sorry, incremental ' + str(len(iscope)) + '-arity cost functions not implemented yet in toulbar2.')
            
            self.CFN.wcsp.postNullaryConstraint(mincost)
            idx = self.CFN.wcsp.postNaryConstraintBegin(iscope, tb2.MAX_COST if isinf(defcost) else int((defcost - mincost) * 10 ** tb2.option.decimalPoint), len(tcosts))
            for i, tuple in enumerate(tuples):
                self.CFN.wcsp.postNaryConstraintTuple(idx, [self.CFN.wcsp.toValue(iscope[x], self.CFN.wcsp.toIndex(iscope[x], v)) for x,v in enumerate(tuple)], tb2.MAX_COST if isinf(tcosts[i]) else int((tcosts[i] - mincost) * 10 ** tb2.option.decimalPoint))
            self.CFN.wcsp.postNaryConstraintEnd(idx)
        
    def AddLinearConstraint(self, coefs, scope, operand = '==', rightcoef = 0):
        """AddLinearConstraint creates a linear constraint with integer coefficients.
        The scope corresponds to the variables involved in the left part of the constraint. 
        All variables must belong to the left part (change their coefficient sign if they are originally in the right part). 
        All constant terms must belong to the rigt part.
        
        Args:
            coefs (list or int): array of integer coefficients associated to the left-part variables (or the same integer coefficient is applied to all variables).
            scope (list): variables involved in the left part of the constraint. A variable can be represented by its name (str) or its index (int).
            operand (str): can be either '==' or '<=' or '<' or '>=' or '>'.
            rightcoef (int): constant term in the right part.
           
        Example:
            AddLinearConstraint([1,1,-2], [x,y,z], '==', -1) encodes x + y -2z = -1. 

        """
        if (isinstance(coefs, int)):
            coefs = [coefs for v in scope]
        assert(len(coefs) == len(scope))
        if operand != '>=' and operand != '>' and operand != '<=' and operand != '<' and operand != '==':
            raise RuntimeError("Unknown operand in AddLinearConstraint: "+str(operand))
        iscope = []
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices.get(v, -1)
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v) 
        sscope = set(iscope)
        if len(iscope) != len(sscope):
            coefs_ = []
            iscope_ = []
            sscope = set()
            index = {}
            for i, v in enumerate(iscope):
                if v in sscope:
                    coefs_[index[v]] += coefs[i]
                else:
                	index[v] = len(iscope_)
                	coefs_.append(coefs[i])
                	iscope_.append(v)
                	sscope.add(v)
            assert(len(iscope_) == len(sscope))
            coefs = coefs_
            iscope = iscope_
        if operand == '>=' or operand == '>' or operand == '==':
            params = str((rightcoef + 1) if (operand == '>') else rightcoef)
            for i,v in enumerate(iscope):
                params += ' ' + str(self.CFN.wcsp.getDomainInitSize(v))
                for valindex in range(self.CFN.wcsp.getDomainInitSize(v)):
                    params += ' ' + str(self.CFN.wcsp.toValue(v, valindex)) + ' ' + str(coefs[i] * self.CFN.wcsp.toValue(v, valindex))
            self.CFN.wcsp.postKnapsackConstraint(iscope, params, kp = True)
        if operand == '<=' or operand == '<' or operand == '==':
            params = str((-rightcoef + 1) if (operand == '<') else -rightcoef)
            for i,v in enumerate(iscope):
                params += ' ' + str(self.CFN.wcsp.getDomainInitSize(v))
                for valindex in range(self.CFN.wcsp.getDomainInitSize(v)):
                    params += ' ' + str(self.CFN.wcsp.toValue(v, valindex)) + ' ' + str(-coefs[i] * self.CFN.wcsp.toValue(v, valindex))
            self.CFN.wcsp.postKnapsackConstraint(iscope, params, kp = True)

    def AddSumConstraint(self, scope, operand = '==', rightcoef = 0):
        """AddSumConstraint creates a linear constraint with unit coefficients.
        The scope corresponds to the variables involved in the left part of the constraint.
        
        Args:
            scope (list): variables involved in the left part of the constraint. A variable can be represented by its name (str) or its index (int).
            operand (str): can be either '==' or '<=' or '<' or '>=' or '>'.
            rightcoef (int): constant term in the right part.
           
        Example:
            AddSumConstraint([x,y,z], '<', 3) encodes x + y + z < 3. 

        """
        self.AddLinearConstraint(1, scope, operand, rightcoef)

    def AddGeneralizedLinearConstraint(self, tuples, operand = '==', rightcoef = 0):
        """AddGeneralizedLinearConstraint creates a linear constraint with integer coefficients associated to domain values. 
        The scope implicitely corresponds to the variables involved in the tuples. Missing domain values have an implicit zero coefficient. 
        All constant terms must belong to the right part.
        
        Args:
            tuples (list): array of triplets (variable, domain value, coefficient) in the left part of the constraint.
            operand (str): can be either '==' or '<=' or '<' or '>=' or '>'.
            rightcoef (int): constant term in the right part.
           
        Example:
            AddGeneralizedLinearConstraint([('x',1,1),('y',1,1),('z',0,2)], '==', 1) encodes (x==1) + (y==1) + 2*(z==0) = 1 assuming 0/1 variables and (x==u) is equal to 1 if value u is assigned to x else equal to 0. 

        """
        sscope = set()
        scope = []
        for (v, val, coef) in tuples:
            if v not in sscope:
                sscope.add(v)
                scope.append(v)
        if operand != '>=' and operand != '>' and operand != '<=' and operand != '<' and operand != '==':
            raise RuntimeError("Unknown operand in AddGeneralizedLinearConstraint: "+str(operand))
        iscope = []
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices.get(v, -1)
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v) 

        if operand == '>=' or operand == '>' or operand == '==':
            params = str((rightcoef + 1) if (operand == '>') else rightcoef)
            for v in iscope:
                vtuples = [[str(val), str(coef)] for (var, val, coef) in tuples if (isinstance(var, str) and self.VariableIndices[var]==v) or (not isinstance(var, str) and var==v)]
                params += ' ' + str(len(vtuples))
                for e in vtuples:
                    params += ' ' + str(e[0]) + ' ' + str(e[1])
            self.CFN.wcsp.postKnapsackConstraint(iscope, params, kp = True)
        if operand == '<=' or operand == '<' or operand == '==':
            params = str((-rightcoef + 1) if (operand == '<') else -rightcoef)
            for v in iscope:
                vtuples = [[str(val), str(-coef)] for (var, val, coef) in tuples if (isinstance(var, str) and self.VariableIndices[var]==v) or (not isinstance(var, str) and var==v)]
                params += ' ' + str(len(vtuples))
                for e in vtuples:
                    params += ' ' + str(e[0]) + ' ' + str(e[1])
            self.CFN.wcsp.postKnapsackConstraint(iscope, params, kp = True)

    def AddAllDifferent(self, scope, encoding = 'binary', excepted = None, incremental = False):
        """Add AllDifferent hard global constraint.
        
        Args:
            scope (list): input variables of the function. A variable can be represented by its name (str) or its index (int).
            encoding (str): encoding used to represent AllDifferent (available choices are 'binary' or 'salldiff' or 'salldiffdp' or 'salldiffkp' or 'walldiff').
            excepted (None or list): list of excepted domain values which can be taken by any variable without violating the constraint.
            incremental (bool): if True then the constraint is backtrackable (i.e., it disappears when restoring at a lower depth, see Store/Restore).
            
        """
        if incremental and encoding != 'binary':
            raise RuntimeError("Implementation of AllDifferent constraint requires 'binary' encoding in incremental mode!")
        if excepted is not None and encoding != 'binary':
            raise RuntimeError("Excepted domain values in AllDifferent constraint requires 'binary' encoding!")
        sscope = set(scope)
        if len(scope) != len(sscope):
            raise RuntimeError("Duplicate variable in scope:"+str(scope))
        iscope = []
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices.get(v, -1)
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v)
        if (len(iscope) >= 2):
            if (encoding=='binary'):
                for i in range(len(iscope)):
                    for j in range(i+1, len(iscope)):
                        costs = [(0 if (self.CFN.wcsp.toValue(iscope[i], a) != self.CFN.wcsp.toValue(iscope[j], b) or (excepted and ((self.CFN.wcsp.toValue(iscope[i], a) in excepted) or (self.CFN.wcsp.toValue(iscope[j], b) in excepted)))) else self.Top) for a in range(self.CFN.wcsp.getDomainInitSize(iscope[i])) for b in range(self.CFN.wcsp.getDomainInitSize(iscope[j]))]
                        self.CFN.wcsp.postBinaryConstraint(iscope[i], iscope[j], costs, incremental)
            elif (encoding=='salldiff'):
                self.CFN.wcsp.postWAllDiff(iscope, "var", "flow", tb2.MAX_COST);
            elif (encoding=='salldiffdp'):
                self.CFN.wcsp.postWAllDiff(iscope, "var", "DAG", tb2.MAX_COST);
            elif (encoding=='salldiffkp'):
                self.CFN.wcsp.postWAllDiff(iscope, "hard", "knapsack", tb2.MAX_COST);
            elif (encoding=='salldiffeq'):
                self.CFN.wcsp.postWAllDiff(iscope, "hardeq", "knapsack", tb2.MAX_COST);
            elif (encoding=='salldiffge'):
                self.CFN.wcsp.postWAllDiff(iscope, "hardge", "knapsack", tb2.MAX_COST);
            elif (encoding=='walldiff'):
                self.CFN.wcsp.postWAllDiff(iscope, "hard", "network", tb2.MAX_COST);
            else:
                raise RuntimeError("Unknown encoding for AllDifferent: "+encoding)

    def AddGlobalFunction(self, scope, gcname, *parameters):
        """AddGlobalFunction creates a soft global cost function. 
        
        Args:
            scope (list): input variables of the function. A variable can be represented by its name (str) or its index (int).
            gcname (str): name of the global cost function (see toulbar2 user documentation).
            parameters (list): list of parameters (str or int) for this global cost function.
            
        Example:
            AddGlobalFunction(['x1','x2','x3','x4'], 'wamong', 'hard', 1000, 2, 1, 2, 1, 3) encodes a hard among constraint satisfied iff values {1,2} are assigned to the given variables at least once and at most 3 times, otherwise it returns a cost of 1000.
        """
        sscope = set(scope)
        if len(scope) != len(sscope):
            raise RuntimeError("Duplicate variable in scope:"+str(scope))
        iscope = []
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices.get(v, -1)
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v)
        params = str(list(parameters))[1:-1].replace(',','').replace('\'','')
        self.CFN.wcsp.postGlobalFunction(iscope, gcname, params)

    def AddWeightedCSPConstraint(self, problem, lb, ub, duplicateHard = False, strongDuality = False):
        """AddWeightedCSPConstraint creates a hard global constraint on the cost of an input weighted constraint satisfaction problem such that its valid solutions must have a cost value in [lb,ub[. 
        
        Args:
            problem (CFN): input problem.
            lb (decimal cost): any valid solution in the input problem must have a cost greater than or equal to lb.
            ub (decimal cost): any valid solution in the input problem must have a cost strictly less than ub.
            duplicateHard (bool): if True then it assumes any forbidden tuple in the original input problem is also forbidden by another constraint in the main model (you must duplicate any hard constraints in your input model into the main model).
            strongDuality (bool): if True then it assumes the propagation is complete when all channeling variables in the scope are assigned and the semantic of the constraint enforces that the optimum and ONLY the optimum on the remaining variables is between lb and ub.
            
        Note:
            If a variable in the input problem does not exist in the current problem (with the same name), it is automatically added.
            
        Example:
            m=tb2.CFN(); m.Read("master.cfn");s=tb2.CFN();s.Read("slave.cfn");m.AddWeightedCSPConstraint(s, lb, ub);m.Solve()
        """
        iscope = []
        for i, v in enumerate(problem.VariableNames):
            if isinstance(v, str):
                vname = v
                v = self.VariableIndices.get(vname, -1)
                if (v < 0 or v >= len(self.VariableNames)):
                    v = self.AddVariable(vname, [(problem.CFN.wcsp.getValueName(i, value) if len(problem.CFN.wcsp.getValueName(i, value)) > 0 else value) for value in problem.Domain(vname)])
            if (v < 0 or v >= len(self.VariableNames)):
                raise RuntimeError("Out of range variable index:"+str(v))
            iscope.append(v)
        multicfn = MultiCFN()
        multicfn.PushCFN(problem, -1)
        negproblem = CFN(vac = self.Option.vac, seed = self.Option.seed, verbose = self.Option.verbose, init = False)
        negproblem.InitFromMultiCFN(multicfn)
        negproblem.UpdateUB(1. - problem.GetLB())
        # keep alive both problem and negproblem
        self.InternalCFNs.append(problem)
        self.InternalCFNs.append(negproblem)
        self.CFN.wcsp.postWeightedCSPConstraint(iscope, problem.CFN.wcsp, negproblem.CFN.wcsp, problem.CFN.wcsp.DoubletoCost(lb), problem.CFN.wcsp.DoubletoCost(ub), duplicateHard, strongDuality)
        
    def Read(self, filename):
        """Read reads the problem from a file.

        Args:
            filename (str): problem filename.

        """
        self.CFN.read(filename)
        self.VariableIndices = {}
        self.VariableNames = []
        self.Variables = {}
        for i in range(self.CFN.wcsp.numberOfVariables()):
            name = self.CFN.wcsp.getName(i)
            self.VariableIndices[name] = i
            self.VariableNames.append(name)
            self.Variables[name] = self.Domain(name)

    def Parse(self, certificate):
        """Parse performs a list of elementary reduction operations on domains of variables.

        Args:
            certificate (str): a string composed of a list of operations on domains, each operation in the form ',varIndex[=#<>]value' 
                               where varIndex (int) is the index of a variable as returned by AddVariable and value (int) is a domain value
                               (comma is mandatory even for the first operation, add no space).
                               Possible operations are: assign ('='), remove ('#'), decrease maximum value ('<'), increase minimum value ('>').
                               
        Example:
            Parse(',0=1,1=1,2#0'): assigns the first and second variable to value 1 and remove value 0 from the third variable. 
            
        """
        self.CFN.parse_solution(certificate, False if self.configuration else True)    # WARNING! False: do not reuse certificate in future searches used by structure learning evaluation procedure!

    def Dump(self, filename):
        """Dump outputs the problem in a file (without doing any preprocessing).

        Args:
            filename (str): problem filename. The suffix must be '.wcsp' or '.cfn' to select in which format to save the problem.

        """
        if self.UbInit is not None:
            self.CFN.wcsp.updateDUb(self.UbInit)
        if '.wcsp' in filename:
            if self.CFN.wcsp.getNegativeLb() > 0 or tb2.option.decimalPoint != 0:
                print('Warning! Problem optimum has been' + (' multiplied by ' + str(10 ** tb2.option.decimalPoint) if tb2.option.decimalPoint != 0 else '') + (' and' if self.CFN.wcsp.getNegativeLb() > 0 and tb2.option.decimalPoint != 0 else '') + (' shifted by ' + str(self.CFN.wcsp.getNegativeLb()) if self.CFN.wcsp.getNegativeLb() > 0 else '') + ' in wcp format')
            self.CFN.dump_wcsp(filename, True, 1)
        elif '.cfn' in filename:
            self.CFN.dump_wcsp(filename, True, 2)
        else:
            print('Error unknown format!')

    def Print(self):
        """Print prints the content of the CFN (variables, cost functions).

        """

        self.CFN.wcsp.print()

    def GetNbVars(self):
        """GetNbVars returns the number of variables.

        Returns:
            Number of variables (int).

        """
        return self.CFN.wcsp.numberOfVariables()

    def Domain(self, var):
        """Domain returns the current domain of a given variable.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.

        Returns:
            List of domain values (list).

        """
        return self.CFN.wcsp.getEnumDomain(self.VariableIndices[var] if isinstance(var, str) else var)

    def GetDomainInitSize(self, var):
        """GetDomainInitSize returns the initial domain size of a given variable.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.

        Returns:
            Initial domain size (int). See also GetValue, GetValueName, and GetValueIndex.

        """
        return self.CFN.wcsp.getDomainInitSize(self.VariableIndices[var] if isinstance(var, str) else var)

    def GetValue(self, var, index):
        """GetValue returns the value at position index in the initial domain of a given variable.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.
            index (int): index of the value in the initial domain of the variable (see also GetDomainInitSize).

        Returns:
            domain value (int). In symbolic domains, the returned value is equal to index (see note in AddVariable).

        """
        return self.CFN.wcsp.toValue(self.VariableIndices[var] if isinstance(var, str) else var, index)

    def GetValueName(self, var, index):
        """GetValueName returns the symbolic value at position index in the initial domain of a given variable.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.
            index (int): index of the value in the initial domain of the variable (see also GetDomainInitSize).

        Returns:
            symbolic name corresponding to a domain value (str).

        """
        return self.CFN.wcsp.getValueName(self.VariableIndices[var] if isinstance(var, str) else var, self.GetValue(var, index))

    def GetValueIndex(self, var, value):
        """GetValueIndex returns the position index in the initial domain of a given variable which corresponds to the given value.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.
            value (int|str): domain value (or its symbolic name).

        Returns:
            Index of the value in the initial domain of the variable (int).

        """
        return self.CFN.wcsp.toIndex(self.VariableIndices[var] if isinstance(var, str) else var, value)

    def GetNbConstrs(self):
        """GetNbConstrs returns the number of non-unary cost functions.

        Returns:
            Number of non-unary cost functions (int).

        """
        return self.CFN.wcsp.numberOfConstraints()
 
    def GetLB(self):
        """GetLB returns the current problem lower bound.

        Returns:
            Current lower bound (decimal cost).

        """
        return self.CFN.wcsp.getDDualBound()

    def GetUB(self):
        """GetUB returns the initial upper bound.

        Returns:
            Current initial upper bound (decimal cost).

        """
        return self.CFN.wcsp.getDPrimalBound()

    # use only for decreasing current upper bound
    def UpdateUB(self, cost):
        """UpdateUB decreases the initial upper bound to a given value. Does nothing if this value is greater than the current upper bound.

        Args:
            cost (decimal cost): new initial upper bound.
            
        Warning:
            This operation might generate a Contradiction if the new upper bound is lower than or equal to the problem lower bound.
            
        """
        self.CFN.wcsp.updateDUb(cost)
        self.CFN.wcsp.enforceUb()   # this might generate a Contradiction exception

    def GetNbNodes(self):
        """GetNbNodes returns the number of search nodes explored so far.

        Returns:
            Current number of search nodes (int).

        """
        return self.CFN.getNbNodes()

    def GetNbBacktracks(self):
        """GetNbBacktracks returns the number of backtracks done so far.

        Returns:
            Current number of backtracks (int).

        """
        return self.CFN.getNbBacktracks()

    def GetSolutions(self):
        """GetSolutions returns all the solutions found so far with their associated costs.

        Returns:
            List of pairs (decimal cost, solution) where a solution is a list of domain values.

        """
        return self.CFN.solutions()

    def GetDDualBound(self):
        """GetDDualBound returns the global problem lower bound in minimization (resp. upper bound in maximization) found after doing an incomplete search with Solve.

        Returns:
            Global lower bound (decimal cost).

        """
        return self.CFN.getDDualBound()

    def GetName(self):
        """GetName get the name of the CFN.

        Returns:
            Name of the CFN (string).
        """

        return self.CFN.wcsp.getName(name)

    def SetName(self, name):
        """SetName set the name of the CFN.

        Args:
            name (str): the new name of the CFN.

        """

        self.CFN.wcsp.setName(name)

        return

    def NoPreprocessing(self):
        """NoPreprocessing deactivates most preprocessing methods.

        """
        tb2.option.elimDegree = -1
        tb2.option.elimDegree_preprocessing = -1
        tb2.option.preprocessTernaryRPC = 0
        tb2.option.preprocessFunctional = 0
        tb2.option.costfuncSeparate = False
        tb2.option.preprocessNary = 0
        tb2.option.DEE = 0
        tb2.option.MSTDAC = False
        tb2.option.trwsAccuracy = -1
        
    # non-incremental solving method
    def Solve(self, showSolutions = 0, allSolutions = 0, diversityBound = 0, timeLimit = 0, writeSolution = ''):
        """Solve solves the problem (i.e., finds its optimum and proves optimality). It can also enumerate (diverse) solutions depending on the arguments.

        Args:
            showSolutions (int): prints solution(s) found (0: show nothing, 1: domain values, 2: variable names with their assigned values, 
                                                               3: variable and value names).
            allSolutions (int): if non-zero, enumerates all the solutions with a cost strictly better than the initial upper bound
                                    until a given limit on the number of solutions is reached.
            diversityBound (int): if non-zero, finds a greedy sequence of diverse solutions where a solution in the list is optimal
                                      such that it also has a Hamming-distance from the previously found solutions greater than a given bound.
                                      The number of diverse solutions is bounded by the argument value of allSolutions.
            timeLimit (int): CPU-time limit in seconds (or 0 if no time limit)
            writeSolution (str): write best solution found in a file using a given file name and using the same format as showSolutions (or write all solutions if allSolutions is non-zero)
            
        Returns:
            The best (or last if enumeration/diversity) solution found as a list of domain values, its associated cost, always strictly lower 
            than the initial upper bound, and the number of solutions found (returned type: tuple(list, decimal cost, int)).
            or None if no solution has been found (the problem has no solution better than the initial upper bound or a search limit occurs).
            See GetSolutions to retrieve of the solutions found so far.
            See GetDDualBound to retrieve of the global problem dual bound found so far.
            
        Warning:
            This operation cannot be called multiple times on the same CFN object (it may modify the problem or its upper bound).

        """
        tb2.option.showSolutions = showSolutions   # show solutions found (0: none, 1: value indexes, 2: value names, 3: variable and value names if available)
        if len(writeSolution) > 0:
            if showSolutions > 0:
                tb2.option.writeSolution(str(showSolutions))
            tb2.option.writeSolution(writeSolution)
        tb2.option.allSolutions = allSolutions   # find all solutions up to a given maximum limit (or 0 if searching for the optimum)
        if diversityBound != 0 and allSolutions > 0:
            tb2.option.divNbSol = allSolutions
            tb2.option.divBound = diversityBound
            tb2.option.divMethod = 3
            self.CFN.wcsp.initDivVariables()
        tb2.check()    # checks compatibility between selected options
        self.Limit = None
        if (timeLimit > 0):
            self.CFN.timer(timeLimit)
        if self.UbInit is not None:
            self.CFN.wcsp.updateDUb(self.UbInit)
        self.CFN.wcsp.sortConstraints()
        solved = self.CFN.solve()
        if len(writeSolution) > 0:
            tb2.option.closeSolution()
        if (len(self.CFN.solutions()) > 0):
            if allSolutions > 0:
                return self.CFN.solutions()[-1][1], self.CFN.solutions()[-1][0], len(self.CFN.solutions()) # returns the last solution found
            else:
                return self.CFN.solution(), self.CFN.wcsp.getDPrimalBound(), len(self.CFN.solutions()) # returns the best solution found
        else:
            return None

    # incremental solving: perform initial preprocessing before all future searches, return improved ub
    def SolveFirst(self):
        """SolveFirst performs problem preprocessing before doing incremental solving.

        Returns:
            Initial upper bound (decimal cost), possibly improved by considering a worst-case situation
            based on the sum of maximum finite cost per function plus one.
            or None if the problem has no solution (a contradiction occurs during preprocessing).
            
        Warning:
            This operation must be done at solver depth 0 (see Depth).
        Warning:
            This operation cannot be called multiple times on the same CFN object.

        """
        if self.UbInit is not None:
            self.CFN.wcsp.updateDUb(self.UbInit)
        tb2.check()    # checks compatibility between selected options
        assert(self.Depth() == 0)
        self.Limit = None
        self.CFN.wcsp.sortConstraints()
        ub = self.CFN.wcsp.getUb()
        self.CFN.beginSolve(ub)
        try:
            ub = self.CFN.preprocessing(ub)
        except tb2.Contradiction:
            self.CFN.wcsp.whenContradiction()
            print('Problem has no solution!')
            return None
        return self.CFN.wcsp.Cost2ADCost(ub)

    # incremental solving: change initial upper bound up and down before adding any problem modifications
    def SetUB(self, cost):
        """SetUB resets the initial upper bound to a given value. It should be done before modifying the problem.

        Args:
            cost (decimal cost): new initial upper bound.
            
        """
        icost = self.CFN.wcsp.DoubletoCost(cost)
        self.Limit = None
        self.CFN.timerStop()
        self.CFN.wcsp.setUb(icost)  # must be done after problem loading
        self.CFN.wcsp.initSolutionCost()  # important to notify previous best found solution is no more valid
        self.CFN.wcsp.enforceUb()   # this might generate a Contradiction exception

    # incremental solving: find the next (optimal) solution after a problem modification (see also SetUB)
    def SolveNext(self, showSolutions = 0, timeLimit = 0):
        """SolveNext solves the problem (i.e., finds its optimum and proves optimality). 
        It should be done after calling SolveFirst and modifying the problem if necessary.

        Args:
            showSolutions (int): prints solution(s) found (0: show nothing, 1: domain values, 2: variable names with their assigned values,
                                                               3: variable and value names).  
            timeLimit (int): CPU-time limit in seconds (or 0 if no time limit)

        Returns:
            The best solution found as a list of domain values, its associated cost, always strictly lower 
            than the initial upper bound, and None (returned type: tuple(list, decimal cost, None)).
            or None if no solution has been found (the problem has no solution better than the initial upper bound or a search limit occurs, see Limit).

        """
        tb2.option.showSolutions = showSolutions   # show solutions found (0: none, 1: value indexes, 2: value names, 3: variable and value names if available)
        tb2.check()    # checks compatibility between selected options
        self.Limit = None
        if (timeLimit > 0):
            self.CFN.timer(timeLimit)
        initub = self.CFN.wcsp.getUb()
        initdepth = tb2.store.getDepth()
        self.CFN.beginSolve(initub)
        tb2.option.hbfs = 1     # reinitialize this parameter which can be modified during hybridSolve()
        try:
            try:
                tb2.store.store()
                self.CFN.wcsp.propagate()
                lb, ub = self.CFN.hybridSolve()
            except tb2.Contradiction:
                self.CFN.wcsp.whenContradiction()
        except tb2.SolverOut as e:
            tb2.option.limit = False
            self.Limit = e
        tb2.store.restore(initdepth)
        if self.CFN.wcsp.getSolutionCost() < initub:
            return self.CFN.solution(), self.CFN.wcsp.getDPrimalBound(), None   # warning! None: does not return number of found solutions because it is two slow to retrieve all solutions in python
        else:
            return None

    # The following functions allow user-defined search procedures:
    
    def Depth(self):
        """Depth returns the current solver depth value.

        Returns:
            Current solver depth value (int).

        """
        return tb2.store.getDepth()

    # make a copy (incremental) of the current problem and move to Depth+1
    def Store(self):
        """Store makes a copy (incremental) of the current problem and increases the solver depth by one.

        """
        tb2.store.store()

    # restore previous copy made at a given depth
    def Restore(self, depth):
        """Restore retrieves the copy made at a given solver depth value.

        Args:
            depth (int): solver depth value. It must be lower than the current solver depth.

        """
        tb2.store.restore(depth)

    def Assign(self, var, value):
        """Assign assigns a variable to a domain value.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.
            value (int): domain value.

        """
        self.CFN.wcsp.assign(self.VariableIndices[var] if isinstance(var, str) else var, value)
        self.CFN.wcsp.propagate()

    def MultipleAssign(self, vars, values):
        """MultipleAssign assigns several variables at once.

        Args:
            vars (list): list of indexes or names of variables.
            values (list): list of domain values.

        """
        self.CFN.wcsp.assignLS([self.VariableIndices[var] if isinstance(var, str) else var for var in vars], values, False)
        
    def Remove(self, var, value):
        """Remove removes a value from the domain of a variable.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.
            value (int): domain value.

        """
        self.CFN.wcsp.remove(self.VariableIndices[var] if isinstance(var, str) else var, value)
        self.CFN.wcsp.propagate()
        
    def Increase(self, var, value):
        """Increase removes the first values strictly lower than a given value in the domain of a variable.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.
            value (int): domain value.

        """
        self.CFN.wcsp.increase(self.VariableIndices[var] if isinstance(var, str) else var, value)
        self.CFN.wcsp.propagate()
        
    def Decrease(self, var, value):
        """Decrease removes the last values strictly greater than a given value in the domain of a variable.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.
            value (int): domain value.

        """
        self.CFN.wcsp.decrease(self.VariableIndices[var] if isinstance(var, str) else var, value)
        self.CFN.wcsp.propagate()
        
    def Deconnect(self, var):
        """Deconnect deconnects a variable from the rest of the problem and assigns it to its support value.

        Args:
            var (int|str): variable name or its index as returned by AddVariable.

        """
        varIndexes = []
        varIndexes.append(self.VariableIndices[var] if isinstance(var, str) else var)
        self.MultipleDeconnect(varIndexes)

    def MultipleDeconnect(self, vars):
        """MultipleDeconnect deconnects a set of variables from the rest of the problem and assigns them to their support value.

        Args:
            vars (list): list of indexes or names of variables.

        """
        self.CFN.wcsp.deconnect([self.VariableIndices[var] if isinstance(var, str) else var for var in vars])
        
    def ClearPropagationQueues(self):
        """ClearPropagationQueues resets propagation queues. It should be called when an exception Contradiction occurs.

        """
        self.CFN.wcsp.whenContradiction()

    def InitFromMultiCFN(self, multicfn, vars=[], scopes=[], constrs=[]):
        """InitFromMultiCFN initializes the cfn from a multiCFN instance (linear combination of multiple CFNs).

        Args:
            multicfn (MultiCFN): the instance containing the CFNs.
            vars (list<int|str>): the list of variable indexes to extract the induced graph (if empty then no restriction)
            scopes (list<set<int|str>>): the list of allowed scopes to extract the partial graph (if empty then no restriction)
            constrs (list<int>): the list of allowed cost function indexes (same index as stored in MultiCFN) to extract the partial graph (if empty then no restriction)
            
        Note:
            After beeing initialized, it is possible to add cost functions to the CFN but the upper bound may be inconsistent.
            The problem lower bound of multicfn is exported only if no restrictions are given (or if scopes contains the emptyset).

        """

        multicfn.MultiCFN.makeWeightedCSP(self.CFN.wcsp, set([multicfn.GetVariableIndex(v) if isinstance(v, str) else v for v in vars]), [set([multicfn.GetVariableIndex(v) if isinstance(v, str) else v for v in scope]) for scope in scopes], list(constrs))
        
        self.VariableIndices = {}
        self.VariableNames = []
        self.Variables = {}
        for i in range(self.CFN.wcsp.numberOfVariables()):
            name = self.CFN.wcsp.getName(i)
            self.VariableIndices[name] = i
            self.VariableNames.append(name)
            self.Variables[name] = self.Domain(name)
        
        return

class MultiCFN:
    """pytoulbar2 base class used to combine linearly multiple CFNs. See (InitFromMultiCFN) to extract and solve it.
    
    Members:
        MultiCFN: python interface to C++ class MultiCFN.
            
    Note:
        It is important to set the parameter (resolution) to the same ***nonzero*** value when creating every CFN to be pushed in a MultiCFN object.
    
    """
    def __init__(self):

        self.MultiCFN = tb2.MultiCFN()

        return
    
    def PushCFN(self, CFN, weight=1.0):
        """PushCFN add a CFN to the instance.

        Args:
            CFN (CFN): the new CFN to add.
            weight (float): the initial weight of the CFN in the combination.

        """

        if CFN.UbInit is not None:
            CFN.SetUB(CFN.UbInit) # might throw a contradiction

        # this should be done in the CFN class, but the update occurs only when solving the problem
        # this is because DoubletoCost function depends on the negCost and LB, which may be updated when adding cost functions
        # if CFN.UbInit is not None:
            # CFN.CFN.wcsp.updateDUb(CFN.UbInit)

        self.MultiCFN.push_back(CFN.CFN.wcsp, weight)


    def SetWeight(self, cfn_index, weight):
        """SetWeight set a weight of a CFN.

        Args:
            cfn_index (int): index of the CFN (in addition order).
            weight (float): the new weight of the CFN.

        """

        self.MultiCFN.setWeight(cfn_index, weight)


    def GetVariableIndex(self, name):
        """GetVariableIndex returns the index of the variable in the combined cfn

        Args:
            name (str): name of the variable.

        Returns:
            The index of the variable or -1 if not found (int).

        """

        return self.MultiCFN.getVariableIndex(name)


    def GetSolution(self):
        """GetSolution returns the solution of the combined cfn after being solved.

        Returns:
            The solution of the cfn (dic).

        """

        return self.MultiCFN.getSolution()


    def GetSolutionCosts(self):
        """GetSolutionCosts returns the costs of the combined cfn after being solved.

        Returns:
            The costs of the solution of the cfn (list).

        """

        return self.MultiCFN.getSolutionValues()

    def ApproximateParetoFront(self, first_criterion, first_direction, second_criterion, second_direction, showSolutions = 0, timeLimit = 0, timeLimit_per_solution = 0, max_sol_count = 0):
        """ApproximateParetoFront returns the set of supported solutions of the problem on two criteria (on the convex hull of the non dominated solutions).
        
        Args:
            first_criterion (int): index of the first CFN to optimize.
            first_direction (str): direction of the first criterion: 'min' or 'max'.
            second_criterion (int): index of the second CFN to optimize.
            second_direction (str): direction of the second criterion: 'min' or 'max'.
            showSolutions (int): prints all intermediate (dominated and nondominated) solution(s) found (0: show nothing, 1: domain values, 2: variable names with their assigned values,
                                                               3: variable and value names).  
            timeLimit (int): CPU-time limit in seconds for the whole method (0 by default, meaning no time limit)
            timeLimit_per_solution (int): CPU-time limit in seconds for the computation of each solution (0 by default, meaning no time limit)
            max_sol_count (int): limit the maximum number of solutions to compute (0 by default, meaning no limit)


        Returns:
            The non dominated solutions belonging to the convex hull of the pareto front and their costs (tuple).

        """

        optim_dir_first = (tb2.Bicriteria.OptimDir.Min if first_direction == 'min'  else tb2.Bicriteria.OptimDir.Max)
        optim_dir_second = (tb2.Bicriteria.OptimDir.Min if second_direction == 'min' else tb2.Bicriteria.OptimDir.Max)

        # parameters
        tb2.Bicriteria.setGlobalTimeout(timeLimit)
        tb2.Bicriteria.setSolutionTimeout(timeLimit_per_solution)
        tb2.Bicriteria.setMaxSolutionCount(max_sol_count)
        tb2.Bicriteria.setShowSolutions(showSolutions)
        tb2.Bicriteria.setVAC(tb2.option.vac)
        tb2.Bicriteria.setSeed(tb2.option.seed)
        tb2.Bicriteria.setVerbose(tb2.option.verbose)

        tb2.Bicriteria.computeSupportedPoints(self.MultiCFN, first_criterion, second_criterion, (optim_dir_first,optim_dir_second))
        # tb2.Bicriteria.computeNonSupported(self.MultiCFN, (optim_dir_first,optim_dir_second), 500)

        return (tb2.Bicriteria.getSolutions(), tb2.Bicriteria.getPoints()) 


    def Print(self):
        """Print print the content of the multiCFN: variables, cost functions.

        """

        self.MultiCFN.print()
