from agents.RG import RG
from agents.TD import TD

def getAgent(name):
    if name == 'TD':
        return TD

    if name == 'RG':
        return RG

    raise NotImplementedError()
