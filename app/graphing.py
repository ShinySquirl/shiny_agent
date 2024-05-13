# This is for charting Nic's LinkedIn connections
#It plots a basic social graph of Nic's connections grouped by company.
#The next thing that needs to be added is to find secondary connections and find connections between people. 
# So I'm going to use Isabelle's LI data to find mutuals, and map that. 

import pandas as pd
import matplotlib.pyplot as plt
import webbrowser
import janitor
import datetime
import os
from IPython.display import display, HTML
from pyvis.network import Network
import networkx as nx
import webbrowser
import os

def enable_interactive_mode():
    # Enable interactive mode for matplotlib
    plt.ion()

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df = df.drop(columns=['first_name', 'last_name', 'email_address'])  # Drop personal info
    df = df.dropna(subset=['company', 'position'])  # Drop rows with missing company/position
    df['connected_on'] = pd.to_datetime(df['connected_on'], format='%d %b %Y')  # Convert to datetime
    return df

def aggregate_data(df, column_name):
    # Aggregate and sort data
    df_aggregated = df[column_name].value_counts().reset_index()
    df_aggregated.columns = [column_name, 'count']
    df_aggregated = df_aggregated[df_aggregated['count'] >= 5].sort_values(by="count", ascending=False)
    return df_aggregated

def initialize_network_graph():
    nt = Network(notebook=False)
    nt.add_node('Nic', label='Nic', color="#f54242")
    nt.add_node('Isabelle', label='Isabelle', color="#4287f5")
    return nt

def add_nodes_and_edges(nt, df, root_name, edges_added):
    node_sizes = {}  # Dictionary to track the number of connections for each node

    for index, row in df.iterrows():
        company = row['company']
        position = row['position']
        company_node_id = f"{company}"
        position_node_id = f"{position}"

        # Increment connection count for nodes
        node_sizes[company_node_id] = node_sizes.get(company_node_id, 0) + 1
        node_sizes[position_node_id] = node_sizes.get(position_node_id, 0) + 1

        # Add or update nodes with size
        if company_node_id not in nt.node_ids:
            nt.add_node(company_node_id, label=company, color="#4287f5", title=f"Company: {company}", size=node_sizes[company_node_id] * 5)
        else:
            # Re-add the node with the updated size
            nt.add_node(company_node_id, label=company, color="#4287f5", title=f"Company: {company}", size=node_sizes[company_node_id] * 5)

        if position_node_id not in nt.node_ids:
            nt.add_node(position_node_id, label=position, color="#42f554", title=f"Position: {position}", size=node_sizes[position_node_id] * 5)
        else:
            # Re-add the node with the updated size
            nt.add_node(position_node_id, label=position, color="#42f554", title=f"Position: {position}", size=node_sizes[position_node_id] * 5)

        # Add edges
        if (root_name, company_node_id) not in edges_added:
            nt.add_edge(root_name, company_node_id)
            edges_added.add((root_name, company_node_id))
        if (root_name, position_node_id) not in edges_added:
            nt.add_edge(root_name, position_node_id)
            edges_added.add((root_name, position_node_id))
        if (company_node_id, position_node_id) not in edges_added:
            nt.add_edge(company_node_id, position_node_id, title=f"{company} - {position}")
            edges_added.add((company_node_id, position_node_id))s           

def display_graph(nt, g, filename):
    # Generate and display the graph
    nt.from_nx(g)
    nt.show(filename)
    display(HTML(filename))

def open_html_file(filename):
    # Open the HTML file in the default web browser
    html_path = os.path.abspath(filename)
    webbrowser.open('file://' + html_path)

def main():
    nt = initialize_network_graph()
    edges_added = set()  # Initialize an empty set to track added edges
    isabelle_df = load_and_clean_data("/Users/nicbaird/Desktop/agent_hack/isabelle_connections.csv")
    nic_df = load_and_clean_data("/Users/nicbaird/Desktop/agent_hack/nic_connections.csv")
    
    add_nodes_and_edges(nt, nic_df, 'Nic', edges_added)
    add_nodes_and_edges(nt, isabelle_df, 'Isabelle', edges_added)
    
    nt.show("network.html", notebook=False)
    html_path = os.path.abspath("network.html")
    webbrowser.open('file://' + html_path)

if __name__ == "__main__":
    main()


