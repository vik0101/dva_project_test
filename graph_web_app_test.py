import pandas as pd
import numpy as np
import re
# import matplotlib as mpl # optional (here)
# import matplotlib.pyplot as plt
# import seaborn as sns # Optional, will only affect the color of bars and the grid
# import pyodbc
# from datetime import datetime

#Graph
# from itertools import combinations
# import random
import networkx as nx
from pyvis.network import Network
# import plotly.io as pio
# pio.renderers.default = 'notebook'
from IPython.display import display
from IPython.core.display import HTML
# import plotly.graph_objs as go
# from ipywidgets import widgets, interactive
# import imgkit
import ast

df = pd.read_csv('trial_data_dva.csv')
df_small = df[:10].copy()
df_small = df_small[['SourceTag', 'Tags', 'Tags_parsed']]

def extract_tags(text):
    tags = re.findall(r'<(.*?)>', text)
    return tags


def Network_Maker(df):
    df_2 = df.copy()
    df_2['new_tags'] = df_2['Tags'].apply(extract_tags)
    df_2.drop(['Tags'],inplace=True,axis=1)
    df_2.rename({'new_tags':'Tags'},inplace=True,axis=1)
    
    ####Creating dataframes for unique tags and concatenating#####

    ## Get unique values of tags from df
    unique_tags = df_2["Tags"].explode().unique()
    
    # Create an empty list to store the smaller DataFrames
    dfs = []

    # Iterate over each unique tag
    for tag in unique_tags:
        # Create a smaller DataFrame for the current tag
        temp_df = df_2[df_2['Tags'].apply(lambda x: tag in x)]
        temp_df['SourceTag2'] = tag
        

        # Append the smaller DataFrame to the list
        dfs.append(temp_df)
        
        # Remove the tag value from the list in df_2['Tags']
        df_2['Tags'] = df_2['Tags'].apply(lambda x: [i for i in x if i != tag])
    
    # Concatenate the DataFrames vertically into a single DataFrame
    dfs = pd.concat(dfs)
    
    
    # Drop rows where df['Tags'] is equal to df['SourceTag2']
    dfs = dfs[~(dfs['Tags'].apply(len) == 1)]
    
    # Reset the index of the DataFrame
    dfs = dfs.reset_index(drop=True)
    dfs = dfs[['SourceTag2', 'Tags']]
    dfs = dfs.rename(columns={"SourceTag2": "SourceTag"})

    
#     display(dfs)


    exploded_df = dfs.explode('Tags')
    filtered_df = exploded_df[exploded_df['Tags'] != exploded_df['SourceTag']]
    
    ##############data frame creation ends########
    
    #######Network Graph Code begins here##########
    
    data = pd.DataFrame({'source': filtered_df['SourceTag'],
                         'target': filtered_df['Tags']})
    data['target'] = data['target'].astype("string")

    edges = data.groupby(by=['source',"target"]).size().reset_index(name='weight')
    edges = edges.nlargest(n=30,columns=['weight'])

    g = nx.from_pandas_edgelist(edges, 'source','target','weight')
    net = Network(
        notebook=True,
        cdn_resources="in_line",
        bgcolor="#222222",
        font_color="white",
        select_menu=True
        # height="2000px",
        # width="2000px",
    )
    for node in g.nodes:
        node_size = g.degree[node]*2
        net.add_node(node, size=node_size)

    for source, target, weight in g.edges.data('weight'):
        net.add_edge(source,target,value=weight)
    # nodes = list(set([*edges.source, *edges.target]))
        edges_list = edges.values.tolist()


###     Enable physics for draggable nodes
    net.toggle_physics(True)
    return net.show("stackoverflow_network.html")

network_graph_small = Network_Maker(df_small)
network_graph_small