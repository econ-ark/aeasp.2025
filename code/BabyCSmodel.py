'''
This module provides stripped down versions of a consumption-saving model for
teaching purposes.  Some of the functionality of ConsIndShockModel.py is
replicated here.  The baby model is solved in three different ways: by explicit
maximization, by satisfying the first order condition, and by endogenous grid.
It requires the HARK package to be installed in order to run.
'''

import numpy as np
from HARK import AgentType
from HARK.metric import MetricObject
from HARK.rewards import CRRAutility, CRRAutilityP, CRRAutilityP_inv, CRRAutility_inv,\
                CRRAutility_invP
from HARK.utilities import make_grid_exp_mult, plot_funcs, plot_funcs_der
from HARK.distributions import MeanOneLogNormal, combine_indep_dstns
from HARK.interpolation import LinearInterp, CubicInterp
from scipy.optimize import fminbound, brentq
import matplotlib.pyplot as plt

class BabyConsumerSolution(MetricObject):
    '''
    A class for representing one period of the solution to a "baby consumption-
    saving" problem.
    '''
    distance_criteria = ['Cfunc']
    
    def __init__(self,Cfunc=None,Vfunc=None,VpFunc=None):
        if Cfunc is not None:
            self.Cfunc = Cfunc
        if Vfunc is not None:
            self.Vfunc = Vfunc
        if VpFunc is not None:
            self.VpFunc = VpFunc
            
            
class BabyValueFunction(MetricObject):
    '''
    A class for representing a value function with CRRA utility using the pseudo-
    inverse value function.
    '''
    def __init__(self, VnvrsFunc, CRRA):
        self.VnvrsFunc = VnvrsFunc
        self.CRRA = CRRA
        self.u = lambda x : CRRAutility(x, gam=self.CRRA)
        self.uP = lambda x : CRRAutilityP(x, gam=self.CRRA)
        
    def __call__(self,M):
        return self.u(self.VnvrsFunc(M))
    
    def derivative(self,M):
        return self.VnvrsFunc.derivative(M) * self.uP(self.VnvrsFunc(M))
        
            
def solveBabyCSbyMaximization(solution_next,DiscFac,Rfree,CRRA,IncomeDstn,StateGrid):
    '''
    Solves one period of the "baby consumption-saving" model using explicit
    maximization over consumption choices.
    
    Parameters
    ----------
    solution_next : BabyConsumerSolution
        The solution to the next period's problem; should have the attributes
        Vfunc and Cfunc, representing the value and consumption functions.
    DiscFac : float
        Intertemporal discount factor.
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    CRRA : float
        Coefficient of relative risk aversion.
    IncomeDstn : DiscreteDistribution
        Distribution of income received next period.  Has attributes pmf and X.
    StateGrid : np.array
        Array of states at which the consumption-saving problem will be solved.
        Represents values of M_t or cash-on-hand ("market resources").
        
    Returns
    -------
    solution_now : BabyConsumerSolution
        The solution to this period's problem.
    '''
    # Unpack next period's solution and the income distribution, and define the utilty function
    Vfunc_next = solution_next.Vfunc
    IncomeProbs = IncomeDstn.pmv
    IncomeVals  = IncomeDstn.atoms[0]
    u = lambda C : CRRAutility(C, CRRA)
    
    EvalCount = 0
    
    # Initialize the arrays of optimal consumption and value
    MnowArray = np.zeros_like(StateGrid) + np.nan
    CnowArray = np.zeros_like(StateGrid) + np.nan
    VnowArray = np.zeros_like(StateGrid) + np.nan

    # Loop through the states, finding optimal consumption and value at each one
    for j in range(StateGrid.size):
        Mnow = StateGrid[j]

        # Define a temporary function that yields (negative) consumption-conditional payoffs.
        # This is defined *negatively* because we're using a minimizer, not a maximizer.
        def payoffFunc(C):
            Anow = Mnow - C # End-of-period assets
            Mnext = Rfree*Anow + IncomeVals # Next period's market resources M_{t+1}
            Vnext = Vfunc_next(Mnext) # Next period's value
            payoff = u(C) + DiscFac*np.dot(Vnext,IncomeProbs) # Utility plus expected value
            return -payoff
            
        # Search over consumption values to maximize the payoff
        Cmin = 0.000000000001*Mnow
        Cmax = 0.999999999999*Mnow
        Cnow, trash1, trash2, EvalCount_j = fminbound(payoffFunc, Cmin, Cmax, full_output=True)
        EvalCount += EvalCount_j
        
        # Calculate the value of optimal choice and store the results in the arrays
        Vnow = -payoffFunc(Cnow)
        MnowArray[j] = Mnow
        CnowArray[j] = Cnow
        VnowArray[j] = Vnow

    # Construct consumption and value functions for this period
    CfuncNow = LinearInterp(np.insert(MnowArray,0,0.0),np.insert(CnowArray,0,0.0))
    VfuncNow = LinearInterp(MnowArray, VnowArray, lower_extrap = True)
    
    # Make a solution object for this period and return it
    solution_now = BabyConsumerSolution(Cfunc=CfuncNow,Vfunc=VfuncNow)
    return solution_now
    
    
def solveBabyCSbyFirstOrderCondition(solution_next,DiscFac,Rfree,CRRA,IncomeDstn,StateGrid):
    '''
    Solves one period of the "baby consumption-saving" model by satisfying the
    first order condition for optimality (where possible).
    
    Parameters
    ----------
    solution_next : BabyConsumerSolution
        The solution to the next period's problem; should have the attributes
        VpFunc and Cfunc, representing the marginal value and consumption functions.
    DiscFac : float
        Intertemporal discount factor.
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    CRRA : float
        Coefficient of relative risk aversion.
    IncomeDstn : DiscreteDistribution
        Distribution of income received next period.  Has attributes pmf and X.
    StateGrid : np.array
        Array of states at which the consumption-saving problem will be solved.
        Represents values of M_t or cash-on-hand ("market resources").
        
    Returns
    -------
    solution_now : BabyConsumerSolution
        The solution to this period's problem.
    '''
    # Unpack next period's solution and the income distribution, and define the marginal utilty function
    VpFunc_next = solution_next.VpFunc
    IncomeProbs = IncomeDstn.pmf
    IncomeVals  = IncomeDstn.X
    uP = lambda C : CRRAutilityP(C,gam=CRRA)
    
    EvalCount = 0
    
    # Initialize the array of optimal consumption
    MnowArray = np.zeros_like(StateGrid) + np.nan
    CnowArray = np.zeros_like(StateGrid) + np.nan

    # Loop through the states, finding optimal consumption and value at each one
    for j in range(StateGrid.size):
        Mnow = StateGrid[j]

        # Define a temporary function that yields the first order condition, which we want to zero
        def FOCfunc(C):
            Anow = Mnow - C # End-of-period assets
            Mnext = Rfree*Anow + IncomeVals # Next period's market resources
            VpNext = VpFunc_next(Mnext) # Next period's marginal value
            FOC = uP(C) - DiscFac*Rfree*np.dot(VpNext,IncomeProbs) # Marginal utility less expected marginal value
            return FOC
            
        # Search over consumption values to satisfy the first order condition
        if FOCfunc(Mnow) < 0.0: # As long as we want to consume less than all of our resources...
            Cmin = 0.000000000001*Mnow
            Cmax = 0.999999999999*Mnow
            Cnow, Results = brentq(FOCfunc,Cmin,Cmax, full_output=True)
            EvalCount += Results.function_calls
        else:
            Cnow = Mnow
        
        # Store the results in the arrays
        MnowArray[j] = Mnow
        CnowArray[j] = Cnow

    # Construct consumption and marginal value functions for this period
    CfuncNow = LinearInterp(np.insert(MnowArray,0,0.0),np.insert(CnowArray,0,0.0))
    VpFuncNow = lambda M : uP(CfuncNow(M)) # Use envelope condition to define marginal value
    
    # Make a solution object for this period and return it
    solution_now = BabyConsumerSolution(Cfunc=CfuncNow,VpFunc=VpFuncNow)
    return solution_now
    
    
def solveBabyCSbyEndogenousGrid(solution_next,DiscFac,Rfree,CRRA,IncomeDstn,StateGrid):
    '''
    Solves one period of the "baby consumption-saving" model by using the endogenous
    grid method to invert the first order condition, obviating any search.
    
    Parameters
    ----------
    solution_next : BabyConsumerSolution
        The solution to the next period's problem; should have the attributes
        VpFunc and Cfunc, representing the marginal value and consumption functions.
    DiscFac : float
        Intertemporal discount factor.
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    CRRA : float
        Coefficient of relative risk aversion.
    IncomeDstn : DiscreteDistribution
        Distribution of income received next period.  Has attributes pmf and X.
    StateGrid : np.array
        Array of states at which the consumption-saving problem will be solved.
        Represents values of A_t or end-of-period assets.
        
    Returns
    -------
    solution_now : BabyConsumerSolution
        The solution to this period's problem.
    '''
    # Unpack next period's solution and the income distribution, and define the (inverse) marginal utilty function
    VpFunc_next = solution_next.VpFunc
    IncomeProbs = IncomeDstn.pmf
    IncomeVals  = IncomeDstn.X
    uP = lambda C : CRRAutilityP(C,gam=CRRA)
    uPinv = lambda C : CRRAutilityP_inv(C,gam=CRRA)
    
    EvalCount = 0 # Count the number of numeric integrations we need to perform
    
    # Initialize the array of optimal consumption
    StateGridTemp = np.insert(StateGrid,0,0.0) # Add a point at A_t = 0.
    MnowArray = np.zeros_like(StateGridTemp) + np.nan
    CnowArray = np.zeros_like(StateGridTemp) + np.nan

    # Loop through the states, finding optimal consumption and value at each one
    for j in range(StateGridTemp.size):
        Anow = StateGridTemp[j]
        Mnext = Rfree*Anow + IncomeVals # Next period's market resources
        VpNext = VpFunc_next(Mnext) # Next period's marginal value
        EndOfPeriodVp = DiscFac*Rfree*np.dot(VpNext,IncomeProbs) # Marginal value of end-of-period assets
        Cnow = uPinv(EndOfPeriodVp) # Invert the first order condition to find how much we must have *just consumed*
        Mnow = Anow + Cnow # Find beginning of period market resources using end-of-period assets and consumption
        
        # Store the results in the arrays
        MnowArray[j] = Mnow
        CnowArray[j] = Cnow
        EvalCount += 1

    # Construct consumption and marginal value functions for this period
    CfuncNow = LinearInterp(np.insert(MnowArray,0,0.0),np.insert(CnowArray,0,0.0))
    VpFuncNow = lambda M : uP(CfuncNow(M)) # Use envelope condition to define marginal value
    
    # Make a solution object for this period and return it
    solution_now = BabyConsumerSolution(Cfunc=CfuncNow,VpFunc=VpFuncNow)
    return solution_now


def solveBabyCSplusVfunc(solution_next,DiscFac,Rfree,CRRA,IncomeDstn,StateGrid):
    '''
    Solves one period of the "baby consumption-saving" model by using the endogenous
    grid method to invert the first order condition, obviating any search.
    
    Parameters
    ----------
    solution_next : BabyConsumerSolution
        The solution to the next period's problem; should have the attributes
        VpFunc and Cfunc, representing the marginal value and consumption functions.
    DiscFac : float
        Intertemporal discount factor.
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    CRRA : float
        Coefficient of relative risk aversion.
    IncomeDstn : DiscreteDistribution
        Distribution of income received next period.  Has attributes pmf and X.
    StateGrid : np.array
        Array of states at which the consumption-saving problem will be solved.
        Represents values of A_t or end-of-period assets.
        
    Returns
    -------
    solution_now : BabyConsumerSolution
        The solution to this period's problem.
    '''
    # Unpack next period's solution and the income distribution, and define the (inverse) marginal utilty function
    Vfunc_next = solution_next.Vfunc
    VpFunc_next = solution_next.VpFunc
    IncomeProbs = IncomeDstn.pmf
    IncomeVals  = IncomeDstn.X
    u = lambda C : CRRAutility(C,gam=CRRA)
    uinv = lambda u : CRRAutility_inv(u,gam=CRRA)
    uinvP = lambda u : CRRAutility_invP(u,gam=CRRA)
    uP = lambda C : CRRAutilityP(C,gam=CRRA)
    uPinv = lambda C : CRRAutilityP_inv(C,gam=CRRA)
    
    EvalCount = 0
    
    # Initialize the array of optimal consumption
    StateGridTemp = np.insert(StateGrid,0,0.0) # Add a point at A_t = 0.
    MnowArray = np.zeros_like(StateGridTemp) + np.nan
    CnowArray = np.zeros_like(StateGridTemp) + np.nan
    VnowArray = np.zeros_like(StateGridTemp) + np.nan

    # Loop through the states, finding optimal consumption and value at each one
    for j in range(StateGridTemp.size):
        Anow = StateGridTemp[j]
        Mnext = Rfree*Anow + IncomeVals # Next period's market resources
        VpNext = VpFunc_next(Mnext) # Next period's marginal value
        EndOfPeriodVp = DiscFac*Rfree*np.dot(VpNext,IncomeProbs) # Marginal value of end-of-period assets
        Cnow = uPinv(EndOfPeriodVp) # Invert the first order condition to find how much we must have *just consumed*
        Mnow = Anow + Cnow # Find beginning of period market resources using end-of-period assets and consumption
        
        Vnext = Vfunc_next(Mnext) # Next period's value
        EndOfPeriodV = DiscFac*np.dot(Vnext,IncomeProbs) # Value of end-of-period assets
        EvalCount += 1
        
        # Store the results in the arrays
        MnowArray[j] = Mnow
        CnowArray[j] = Cnow
        VnowArray[j] = u(Cnow) + EndOfPeriodV
        
        # Set the bottom EndOfPrdV value aside (EndOfPrdV(A=0))
        if Anow == 0.:
            EndOfPeriodV_at_0 = EndOfPeriodV

    # Construct consumption and marginal value functions for this period
    CfuncNow = LinearInterp(np.insert(MnowArray,0,0.0),np.insert(CnowArray,0,0.0))
    VpFuncNow = lambda M : uP(CfuncNow(M)) # Use envelope condition to define marginal value
    
    # Augment the value array with extra points on the constrained portion
    MnowExtra = make_grid_exp_mult(0.001, 0.95*MnowArray[0], 10, timestonest=3) # Extra M_t points clustered at bottom
    VnowExtra = u(MnowExtra) + EndOfPeriodV_at_0 # Value on the constrained portion is u(M_t) + EndOfPrdV(A_t=0.0)
    MnowArrayAlt = np.concatenate((MnowExtra,MnowArray)) # Combine the two pieces of the M_t grid
    VnowArray = np.concatenate((VnowExtra,VnowArray)) # Combine the two pieces of the V_t grid
    
    # Four versions of how to construct the value function, selected using VfuncDecurving and VfuncSmoothing
    VfuncSmoothing = True
    VfuncDecurving = True
    if VfuncDecurving and VfuncSmoothing: # Construct the value function using cubic interpolating on the "decurved" value function
        VnvrsArray  = uinv(VnowArray) # "Decurved" or pseudo-inverse value
        VpNowArray  = np.concatenate((uP(MnowExtra),uP(CnowArray))) # Combine the two VpNowArray sections: constrained and unconstrained
        VnvrsParray = VpNowArray*uinvP(VnowArray)
        VnvrsFunc   = CubicInterp(np.insert(MnowArrayAlt,0,0.0),np.insert(VnvrsArray,0,0.0),np.insert(VnvrsParray,0,1.0))
        VfuncNow    = BabyValueFunction(VnvrsFunc, CRRA)
        #VfuncNow    = lambda x : u(VnvrsFunc(x))
    
    elif VfuncSmoothing: # Construct the value function by using cubic interpolation on VnowArray and VpNowArray
        VpExtra = uP(MnowExtra) # Marginal value on constrained portion is u'(C_t) = u'(M_t)
        VpNowArray = np.concatenate((VpExtra,uP(CnowArray))) # Combine the two VpNowArray sections: constrained and unconstrained
        VfuncNow = CubicInterp(MnowArrayAlt,VnowArray,VpNowArray, lower_extrap=True) # Use cubic spline interpolation to make a smooth value function
        
    elif VfuncDecurving: # Construct the value function for this period by "decurving" through the inverse utility function   
        VnvrsArray = uinv(VnowArray) # "Decurved" or pseudo-inverse value
        VnvrsFunc  = LinearInterp(np.insert(MnowArrayAlt,0,0.0),np.insert(VnvrsArray,0,0.0)) # Construct a linear interpolation of Vnvrs
        VfuncNow   = BabyValueFunction(VnvrsFunc, CRRA)
        #VfuncNow   = lambda x : u(VnvrsFunc(x)) # Recurve the pseudo-inverse value function through the utility function
        
    else: # Construct the value function for this period by linearly interpolating on VnowArray
        VfuncNow   = LinearInterp(MnowArrayAlt,VnowArray,lower_extrap=True)
    
    # Make a solution object for this period and return it
    solution_now = BabyConsumerSolution(Cfunc=CfuncNow,VpFunc=VpFuncNow,Vfunc=VfuncNow)
    return solution_now


    
class BabyConsumerType(AgentType):
    '''
    A class for representing an ex ante homogeneous type of consumer in the "baby
    consumption-saving" model.  These consumers have CRRA utility over current
    consumption and discount future utility exponentially.  Their future income
    is subject to transitory shocks, and they can earn gross interest on retained
    assets at a risk free interest factor.
    '''
    def __init__(self, **kwds):
        AgentType.__init__(self, **kwds)
        self.time_vary = []
        self.time_inv = ['DiscFac','Rfree','CRRA']
        self.pseudo_terminal = False
        self.solve_one_period = solveBabyCSbyMaximization
        
    def pre_solve(self):
        '''
        Method that is run automatically when solve() is called.  Creates the
        grid of states, makes the distribution of income shocks, and solves the
        terminal period.
        '''
        self.makeStateGrid()
        self.makeIncomeDstn()
        self.solveTerminal()
        
    def post_solve(self):
        '''
        Store the components of the solution as attributes of self for convenience.
        '''
        self.unpack('Cfunc')
        
        
    def makeStateGrid(self):
        '''
        Uses primitive parameters to construct the attribute StateGrid.
        '''
        if self.ExponentialGrid:
            self.StateGrid = make_grid_exp_mult(ming=self.StateMin, # Minimum value of the state variable
                                         maxg=self.StateMax, # Maximum value of the state variable
                                         ng=self.StateCount, # Number of elements in the vector of states
                                         timestonest=3)      # Exponential nesting factor
        else:
            self.StateGrid = np.linspace(self.StateMin,self.StateMax,self.StateCount)
        self.add_to_time_inv('StateGrid')
        
    def makeIncomeDstn(self):
        '''
        Uses primitive parameters to construct the attribute IncomeDstn.
        '''
        IncomeDstn = MeanOneLogNormal(sigma=self.IncomeStd) # Standard deviation of underlying normal distribution
        IncomeDstnApprox = IncomeDstn.discretize(self.ShkCount) # Discrete approximation with given number of points
        IncomeDstnApprox.atoms[0] *= self.IncomeMean # Shift income values by constant factor
        self.IncomeDstn = IncomeDstnApprox
        self.add_to_time_inv('IncomeDstn')
        
    def solveTerminal(self):
        '''
        Solves the terminal period problem, in which the agent will simply
        consume all available resources.
        '''
        Cfunc_terminal = lambda M : M
        Vfunc_terminal = lambda M : CRRAutility(M,self.CRRA)
        VpFunc_terminal = lambda M : CRRAutilityP(M,self.CRRA)
        self.solution_terminal = BabyConsumerSolution(Cfunc=Cfunc_terminal,
                                                      Vfunc=Vfunc_terminal,
                                                      VpFunc=VpFunc_terminal)
        
        


if __name__ == '__main__':
    from time import time
    
    # Define dictionaries of example parameter values
    baby_dict =  {'DiscFac' : 0.95,
                  'Rfree' : 1.03,
                  'CRRA' : 2.0,
                  'StateMin' : 0.001,
                  'StateMax' : 20.0,
                  'StateCount' : 50,
                  'ExponentialGrid' : True,
                  'ShkCount' : 15,
                  'IncomeStd' : 0.2,
                  'IncomeMean' : 1.0,
                  }

    # Make an example baby type
    BabyType = BabyConsumerType(**baby_dict)
    BabyType.cycles = 1 # How many times does the non-terminal period occur? 0 --> infinity
    
    # Change the solution method
    #BabyType.solve_one_period = solveBabyCSbyMaximization
    #BabyType.solve_one_period = solveBabyCSbyFirstOrderCondition
    #BabyType.solve_one_period = solveBabyCSbyEndogenousGrid
    #BabyType.solve_one_period = solveBabyCSplusVfunc
    #BabyType.solve_one_period = solveBabyCSbyEndogenousGridFAST
    
    # Solve the example baby type
    t_start = time()
    BabyType.solve(False)
    t_end = time()
    print('Solving the baby consumption-saving model took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function in the first period
    print('Consumption function in the first period:')
    plot_funcs(BabyType.Cfunc[0],0.0,10.0)
    
    # Plot the value function (if it exists) in the first period of the "baby" model
    if hasattr(BabyType.solution[0],'Vfunc'):
        print('Value function in first period:')
        plot_funcs(BabyType.solution[0].Vfunc,0.1,10.)
