import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel

from src.utils.rlglue import OffPolicyWrapper

from src.problems.registry import getProblem
from src.utils.errors import MSVE, computeVStar
from src.utils.Collector import Collector

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    sys.exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

collector = Collector()
for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    env = problem.getEnvironment()
    rep = problem.getRepresentation()
    agent = problem.getAgent()

    mu = problem.behavior
    pi = problem.target

    # takes actions according to mu and will pass the agent an importance sampling ratio
    # makes sure the agent only sees the state passed through rep.encode.
    # agent does not see raw state
    agent_wrapper = OffPolicyWrapper(agent, problem.getGamma(), mu, pi, rep.encode)

    X = np.array([
        rep.encode(i) for i in range(env.states + 1)
    ])

    P = env.buildTransitionMatrix(pi)
    R = env.buildAverageReward(pi)
    d = env.getSteadyStateDist(mu)

    v_star = computeVStar(P, R, problem.getGamma())

    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    glue.start()
    for step in range(exp.steps):
        # call agent.step and environment.step
        r, o, a, t = glue.step()

        msve = MSVE(agent.theta, v_star, X, d)
        collector.collect('msve', msve)

        # if terminal state, then restart the interface
        if t:
            glue.start()

    # tell the collector to start a new run
    collector.reset()

# save results to disk
from PyExpUtils.results.backends.csv import saveResults

data = collector.all_data['msve']

permutations = exp.numPermutations()
for run, line in enumerate(data):
    inner_idx = permutations * run + idx
    saveResults(exp, inner_idx, 'msve', line)
