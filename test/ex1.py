import numpy as np
import pandas as pd

data = pd.read_csv('candidate_elimination_dataset.csv')

concept = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):

    specific_h = concepts[0].copy()

    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):

        if target[i] == 'yes':

            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == 'no':

            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = '?'
                else:
                    specific_h[x] = '?'

        
        print("\nSteps of Candidate Elimination Algorithm", i + 1)
        print(specific_h)
        print(general_h)

    empty_row = ['?'] * len(specific_h)

    indices = [i for i, val in enumerate(general_h) if val == empty_row]
    for i in indices:
        general_h.remove(i)
    
    return specific_h, general_h


