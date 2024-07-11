import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(root_dir, 'packages', 'python'))
from taxonomy_tools import helm_data as txm_helm_data
from taxonomy_tools import utils as txm_utils
import pandas as pd
import os

import argparse


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Program that takes two file paths")

    # Add arguments for the paths
    parser.add_argument(
        "--taxonomy", "-t", type=str, required=True, help="Path to the taxonomy file"
    )
    parser.add_argument(
        "--data",
        "-d",
        action="append",
        type=str,
        help="Path (or multiple paths) to the HELM data or any other custom dataset.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the resulting graphs.",
    )

    # Parse arguments from the command line
    args = parser.parse_args()
    TAXONOMY_PATH = args.taxonomy
    HELM_RESULTS_PATHS = args.data
    OUTPUT_PATH = args.output

    # Get taxonomy name
    taxonomy_name = os.path.basename(TAXONOMY_PATH).split('.')[0]
    print('Processing taxonomy: %s'%taxonomy_name)

    # Load taxonomy
    taxonomy_graph, labels_graph, undefined_edges, measurable_edges = (
        txm_utils.load_taxonomy(TAXONOMY_PATH, return_all=True, verbose=True)
    )

    # Get all the required datasets from the taxonomy graph
    datasets_list = txm_utils.get_taxonomy_datasets(taxonomy_graph)

    # Read all the required data from HELM
    helm_samples_dict = dict()
    for data_path in HELM_RESULTS_PATHS:
        helm_samples_dict = txm_helm_data.read_helm_data(
            data_path, datasets_list, current_dict=helm_samples_dict, verbose=True
        )

    # Filter for models that were tested on ALL datasets
    helm_samples_fullytested_dict = txm_utils.filter_for_full_samples(helm_samples_dict)

    # Create taxonomy datasets metrics dataframe
    datasets_data_df = txm_utils.get_taxonomy_datasets_metrics_dataframe(
        helm_samples_fullytested_dict
    )
    # Save
    datasets_data_df.to_csv(
        os.path.join(
            OUTPUT_PATH,
            "%s" % taxonomy_name + "_dataset_metrics.csv",
        )
    )

    # Create taxonomy nodes metrics dataframe
    nodes_data_df = txm_utils.get_taxonomy_datasets_node_dataframe(
        helm_samples_fullytested_dict, taxonomy_graph, verbose=True
    )
    # Save
    nodes_data_df.to_csv(
        os.path.join(
            OUTPUT_PATH,
            "%s" % taxonomy_name + "_nodes_metrics.csv",
        )
    )

    # Get nodes correlations
    correlation_matrix, correlation_matrix_filtered = (
        txm_utils.get_taxonomy_nodes_correlation(
            nodes_data_df, taxonomy_graph, verbose=True
        )
    )
    # Save
    pd.DataFrame(correlation_matrix).to_csv(
        os.path.join(
            OUTPUT_PATH,
            "%s" % taxonomy_name + "_full_corrleation.csv",
        ),
        index=False,
        header=False,
    )
    pd.DataFrame(correlation_matrix_filtered).to_csv(
        os.path.join(
            OUTPUT_PATH,
            "%s" % taxonomy_name
            + "_filtered_corrleation.csv",
        ),
        index=False,
        header=False,
    )

    # Get the unbalanced correlation, using all possible models in each edge
    correlation_matrix_imbalanced = txm_utils.get_taxonomy_per_edge_correlation(
        taxonomy_graph, helm_samples_dict, verbose=True
    )
    # Save
    pd.DataFrame(correlation_matrix_imbalanced).to_csv(
        os.path.join(
            OUTPUT_PATH,
            "%s" % taxonomy_name
            + "_imbalanced_corrleation.csv",
        ),
        index=False,
        header=False,
    )

    ############################################################################
    # ----------- Images
    ############################################################################

    pos = nx.nx_pydot.graphviz_layout(
        taxonomy_graph, prog="dot"
    )  # Choose layout algorithm
    # Draw the graph with desired customizations
    plt.figure(figsize=(15, 6))  # Adjust width and height as desired
    nx.draw_networkx(
        taxonomy_graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="black",
        font_size=8,
    )
    plt.draw()
    plt.savefig(os.path.join(OUTPUT_PATH, "%s" % taxonomy_name + "_taxonomy_graph.png"))

    pos = nx.nx_pydot.graphviz_layout(
        labels_graph, prog="dot"
    )  # Choose layout algorithm
    # Draw the graph with desired customizations
    plt.figure(figsize=(25, 4))  # Adjust width and height as desired
    nx.draw_networkx(
        labels_graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="black",
        font_size=8,
    )
    plt.draw()
    plt.savefig(os.path.join(OUTPUT_PATH, "%s" % taxonomy_name + "_dataset_assignment_graph.png"))

    # mas compacta
    correlation_matrix = nodes_data_df.loc[:, (nodes_data_df != 0).any()].corr()
    # Create a heatmap for visualization
    im=plt.matshow(correlation_matrix, cmap="coolwarm")
    im.set_clim([-1.0,1.0]) 
    # Add colorbar
    plt.colorbar()
    # Set column labels
    plt.xticks(
        range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90
    )
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    # Set title
    plt.title("%s" % TAXONOMY_PATH.split("/")[-1])
    plt.draw()
    plt.savefig(os.path.join(OUTPUT_PATH, "%s" % taxonomy_name + "_taxonomy_correlation_matrix.png"), bbox_inches = 'tight')


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
