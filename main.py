import operator
import math
import random
import functools
import numpy as np
from deap import algorithms, base, creator, gp, tools
from DataGeneration import GenerateDummyData

# Define new functions with error handling

def ProtectedDiv(Left, Right):
    try:
        return Left / Right
    except ZeroDivisionError:
        return 1


def ProtectedLog(Value):
    if Value <= 0:
        return 0
    return math.log(Value)


# Generate data
SampleSize = 100
NoiseSigma = 0.1
DataDict = GenerateDummyData(SampleSize, NoiseSigma)
XValues = DataDict["X"]
YValues = DataDict["Y"]

# Set up Primitive Set
PrimitiveSet = gp.PrimitiveSet("MAIN", 1)
PrimitiveSet.addPrimitive(operator.add, 2)
PrimitiveSet.addPrimitive(operator.sub, 2)
PrimitiveSet.addPrimitive(operator.mul, 2)
PrimitiveSet.addPrimitive(ProtectedDiv, 2)
PrimitiveSet.addPrimitive(math.sin, 1)
PrimitiveSet.addPrimitive(math.cos, 1)
PrimitiveSet.addPrimitive(math.exp, 1)
PrimitiveSet.addPrimitive(ProtectedLog, 1)
PrimitiveSet.addEphemeralConstant("Rand", functools.partial(random.uniform, -1, 1))
PrimitiveSet.renameArguments(ARG0="x")

# Create toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

Toolbox = base.Toolbox()
Toolbox.register("expr", gp.genHalfAndHalf, pset=PrimitiveSet, min_=1, max_=3)
Toolbox.register("individual", tools.initIterate, creator.Individual, Toolbox.expr)
Toolbox.register("population", tools.initRepeat, list, Toolbox.individual)
Toolbox.register("compile", gp.compile, pset=PrimitiveSet)

# Evaluation function

def EvaluateIndividual(Individual, Points):
    Func = Toolbox.compile(expr=Individual)
    try:
        Predictions = np.array([Func(x) for x in Points])
    except (OverflowError, ValueError):
        return float("inf"),
    Error = ((Predictions - YValues) ** 2).mean()
    return Error,

Toolbox.register("evaluate", EvaluateIndividual, Points=XValues)
Toolbox.register("select", tools.selTournament, tournsize=3)
Toolbox.register("mate", gp.cxOnePoint)
Toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
Toolbox.register("mutate", gp.mutUniform, expr=Toolbox.expr_mut, pset=PrimitiveSet)

Toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
Toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Genetic Programming parameters
PopulationSize = 300
Generations = 40
CrossoverProb = 0.5
MutationProb = 0.2

Population = Toolbox.population(n=PopulationSize)
HallOfFame = tools.HallOfFame(1)
Stats = tools.Statistics(lambda ind: ind.fitness.values)
Stats.register("avg", np.mean)
Stats.register("min", np.min)

Population, Log = algorithms.eaSimple(
    Population,
    Toolbox,
    cxpb=CrossoverProb,
    mutpb=MutationProb,
    ngen=Generations,
    stats=Stats,
    halloffame=HallOfFame,
    verbose=False,
)

Best = HallOfFame[0]
BestFunc = Toolbox.compile(expr=Best)
Predictions = np.array([BestFunc(x) for x in XValues])
BestMse = ((Predictions - YValues) ** 2).mean()

print("Best Equation:", Best)
print("Mean Squared Error:", BestMse)
