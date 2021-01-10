import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from PyExpPlotting.learning_curves import plotBest
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpPlotting.defaults import PaperDimensions

from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults

BlogDimensions = PaperDimensions(
    columns=1,
    column_width=2,
    text_width=2,
)

setDefaultConference(BlogDimensions)

colors = {
    'TD': 'blue',
    'RG': 'black',
}

if __name__ == "__main__":
    f, ax = plt.subplots(1)

    exp_paths = sys.argv[1:]

    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)

        results = loadResults(exp, 'msve.csv')

        label = exp.agent
        color = colors[label]

        plotBest(results, ax, {
            'prefer': 'small',
            'color': color,
            'label': label,
        })

    ax.set_xlabel('Steps')
    ax.set_ylabel('MSVE')

    save('plots/', 'learning_curve')
