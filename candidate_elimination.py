import pandas as pd

def candidate_elimination(df: pd.DataFrame):
    """
    if output is yes -> check for empty specific_h if emty 
                            then assign with current row data
                     -> otherwise compare specific_h and current row data if it is different, 
                            then replace with "?" 
                                    keep the remaining same

    if output is no -> iterate each if specific_h[i] != current row data[i] then 
                        make the general_h[i][i] = specific_h (which is different form the current row)
    """

    n = df.columns.size - 1

    specific_h = []
    general_h = [["?" for i in range(n)] for i in range(n)]

    for i, row in df.iterrows():

        output = row.iloc[-1] 
        """getting output means a last column"""
        data = row.iloc[:len(row)-1].tolist()
        """data which is except the last column"""


        if (output == "yes"):

            if not specific_h:
                specific_h = data
                continue

            specific_h = [specific_h[i] if specific_h[i] == data[i]  else "?" for i in range(n)]

        else: 

            for x in range(n):
                if specific_h[x] != data[x]:
                    general_h[x][x] = specific_h[x]

    
    general_h_without_duplicate = []
    for i in general_h:
        for j in i:
            if j != "?":
                general_h_without_duplicate.append(i)

    return specific_h, general_h_without_duplicate
                



df = pd.read_csv("./candidate_elimination_dataset.csv")

specific, general = candidate_elimination(df)

print(specific)
print(general)

""" DATASET
sky,temperature,humid,wind,water,forest,output
sunny,warm,normal,strong,warm,same,yes
sunny,warm,high,strong,warm,same,yes
rainy,cold,high,strong,warm,change,no
sunny,warm,high,strong,cool,change,yes
"""