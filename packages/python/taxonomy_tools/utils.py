import networkx as nx
import numpy as np
import pandas as pd
import collections
from typing import List, Tuple


def load_taxonomy(
    file_path: str, return_all: List = False, verbose: bool = False
) -> any:
    """
    Loads a taxonomy in the graphviz format:

        digraph taxonomy_001 {
            Reasoning -> Deduction;
            ...
            }
        digraph taxonomy_001_labeling {
            Reasoning -> LegalSupport;
            ...
            }

    Returns a networkx representation of it.
    """

    # Load both graphs, taxonomy and labels
    graphs_dict = dict()
    with open(file_path) as f:
        for line in f:
            if len(line) == 0:
                continue

            if "{" in line:
                graph_name = line.split("digraph")[-1].split("{")[0].strip()
                if verbose:
                    print("Found graph : %s" % graph_name)
                graphs_dict[graph_name] = nx.DiGraph()
            elif "}" in line:
                continue
            elif " -> " not in line:
                continue
            else:
                # Get nodes
                from_n = (
                    line.split(" -> ")[0].strip().replace(";", "").replace(":", "---")
                )
                to_n = (
                    line.split(" -> ")[-1].strip().replace(";", "").replace(":", "---")
                )
                # Add to graph (wont be duplicated)
                graphs_dict[graph_name].add_node(from_n)
                graphs_dict[graph_name].add_node(to_n)
                # Add edge
                graphs_dict[graph_name].add_edge(from_n, to_n)

    # Check taxonomy file in correct order and naming convention
    assert len(graphs_dict.keys()) == 2
    taxonomy_name = list(graphs_dict.keys())[0]
    assert taxonomy_name + "_labeling" == list(graphs_dict.keys())[1]

    # Add datasets to nodes in the taxonomy graph using the labels graph
    dataset_correspondency = dict()
    for edge in graphs_dict[taxonomy_name + "_labeling"].edges:
        if edge[0] not in dataset_correspondency.keys():
            dataset_correspondency[edge[0]] = list()
        dataset_correspondency[edge[0]].append(edge[1])
    taxonomy_graph = graphs_dict[taxonomy_name]
    labels_graph = graphs_dict[taxonomy_name + "_labeling"]
    nx.set_node_attributes(taxonomy_graph, dataset_correspondency, name="datasets")

    # Get the measurable edges, those with defined datasets in both nodes
    undefined_edges = list()
    measurable_edges = list()
    for edge in taxonomy_graph.edges:
        if (
            taxonomy_graph.nodes[edge[0]].get("datasets", None) is None
        ) or taxonomy_graph.nodes[edge[1]].get("datasets", None) is None:
            undefined_edges.append(edge)
        else:
            measurable_edges.append(edge)
    if verbose:
        print(
            "%d undefined edges of %d edges (%d are potentially measurable)"
            % (len(undefined_edges), len(taxonomy_graph.edges), len(measurable_edges))
        )

    # Check if the graph contains the same dataset in two nodes that are on the
    # same dependency path

    # Get nodes without outgoing connections
    base_nodes = [
        node for node, out_degree in taxonomy_graph.out_degree() if out_degree == 0
    ]

    def recursive_explore(node_path, dataset_list):
        """
        Given a node path and a list of datasets already assigned, checks if the
        incoming edges contain any of these datasets, if thats the case, it
        throws an error.
        For each incoming edge the function calls itself with the updated dataset
        and node path list. This is repeated until the root of the graph is found,
        which has no incoming edges.
        This works because taxonomies are rather small because they should be
        easily understood by humans.
        """
        # Get the node to analyze, the last from the given path
        node = node_path[-1]
        for edge in taxonomy_graph.in_edges(node):
            if node != edge[1]:
                # We dont care on outgoing edges from the analyzed node.
                continue
            else:
                # Get list of datasets used here
                dataset_list_aux = taxonomy_graph.nodes[edge[0]].get("datasets", [])
                for dataset in dataset_list_aux:
                    if dataset in dataset_list:
                        print("Error in path : ")
                        for node in node_path:
                            print("\t%s" % node_path)
                        raise ValueError(
                            "Detected downstream dataset sharing in node %s with %s on dataset %s"
                            % (node, edge[0], dataset)
                        )
                # Go deeper
                recursive_explore(
                    node_path + [edge[0]], dataset_list + dataset_list_aux
                )
        return

    # For each node, go up and make sure no dataset is shared among its paths up
    for node in base_nodes:
        # Explore path
        recursive_explore([node], taxonomy_graph.nodes[node].get("datasets", []))

    # All ok, return graph
    if return_all:
        return taxonomy_graph, labels_graph, undefined_edges, measurable_edges
    else:
        return taxonomy_graph


def get_taxonomy_datasets_per_node(taxonomy_graph: nx.classes.digraph.DiGraph) -> dict:
    """
    Returns the lists of all datasets assigned to each node
    """
    dataset_correspondency = dict()
    for node in taxonomy_graph.nodes:
        datasets = taxonomy_graph.nodes[node].get("datasets", None)
        if datasets is not None:
            dataset_correspondency[node] = datasets
    return dataset_correspondency


def get_taxonomy_datasets(taxonomy_graph: nx.classes.digraph.DiGraph) -> List:
    """
    Gets a list of unique datasets to be used in the given taxonomy.
    """

    dataset_correspondency = get_taxonomy_datasets_per_node(taxonomy_graph)

    datasets_list = list()
    for val in dataset_correspondency.values():
        for dataset in val:
            if dataset not in datasets_list:
                datasets_list.append(dataset)

    return datasets_list


def filter_for_full_samples(samples_dict: dict) -> dict:
    """
    Given a samples dictionary, containing datasets and the corresponding tested
    models, filters all models that were not tested on ALL the datasets.
    """
    # Keep only the models that were tested on all datasets
    included_dataset_count = dict()
    for dataset in samples_dict.keys():
        for model in samples_dict[dataset].keys():
            if model not in included_dataset_count.keys():
                included_dataset_count[model] = 1
            else:
                included_dataset_count[model] += 1
    # Get the models included in all datasets
    use_models = [
        model
        for model, count in included_dataset_count.items()
        if count == len(samples_dict.keys())
    ]
    use_models.sort()
    samples_fullytested_dict = dict()
    for dataset in samples_dict.keys():
        filtered_dict = {
            k: v for k, v in samples_dict[dataset].items() if k in use_models
        }
        # Also sort
        samples_fullytested_dict[dataset] = dict(sorted(filtered_dict.items()))
    # just check
    for dataset in samples_fullytested_dict.keys():
        assert len(samples_fullytested_dict[dataset].keys()) == len(use_models)

    return samples_fullytested_dict


def get_taxonomy_datasets_metrics_dataframe(samples_dict: dict) -> pd.DataFrame:
    """
    Receives a filtered sample dictionary (only with fully tested models) and
    creates a dataframe containing the taxonomy metrics for each dataset and model
    """

    # Matrix of [nodes x models]
    metrics_data_matrix = np.zeros(
        (
            len(samples_dict.keys()),
            len(samples_dict[list(samples_dict.keys())[0]].keys()),
        )
    )
    metrics_count_matrix = np.ones_like(metrics_data_matrix)
    for idx, key in enumerate(samples_dict.keys()):
        metrics_data_matrix[idx, :] += list(samples_dict[key].values())
        metrics_count_matrix[idx, :] += 1
    # average
    metrics_data_matrix /= metrics_count_matrix

    # Create dataframe
    metric_data_df = pd.DataFrame(
        metrics_data_matrix.T,
        index=list(samples_dict[list(samples_dict.keys())[0]].keys()),
        columns=list(samples_dict.keys()),
    )

    return metric_data_df


def get_taxonomy_datasets_node_dataframe(
    samples_dict: dict,
    taxonomy_graph: nx.classes.digraph.DiGraph,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Receives a filtered sample dictionary (only with fully tested models) and
    the taxonomy graph and creates a dataframe containing the taxonomy metrics
    for each taxonomy node and model.
    """

    # Get models names
    use_models = list(samples_dict.values())[0]

    # Matrix of [nodes x models]
    data_matrix = np.zeros((len(taxonomy_graph.nodes), len(use_models)))
    count_matrix = np.ones_like(data_matrix)
    node_names = list(taxonomy_graph.nodes)
    for idx, node in enumerate(node_names):
        node_dataset_list = taxonomy_graph.nodes[node].get("datasets", None)
        if node_dataset_list is None:
            print("No dataset for node : %s" % node)

        else:
            # get the datasets data
            for dataset in taxonomy_graph.nodes[node]["datasets"]:
                values_dict = samples_dict.get(dataset, None)
                # Now fill matrix, data is already sorted
                if values_dict is None:
                    print("No values found for dataset : %s" % dataset)
                else:
                    data_matrix[idx, :] += list(values_dict.values())
                    count_matrix[idx, :] += 1
    # average
    data_matrix /= count_matrix

    data_df = pd.DataFrame(data_matrix.T, index=use_models, columns=node_names)
    return data_df


def get_taxonomy_nodes_correlation(
    data_df: pd.DataFrame,
    taxonomy_graph: nx.classes.digraph.DiGraph,
    verbose: bool = False,
) -> Tuple[np.array, np.array]:
    """
    Using the taxonomy nodes data and the graph calculates the full correlations
    matrix and also a version with only values where valid edges are defined.
    """

    # calculate correlation
    correlation_matrix = data_df.corr()

    # Filter correlation matrix for edges positions only
    correlation_matrix_filtered = np.zeros_like(correlation_matrix.values)
    nodes_array = np.array(taxonomy_graph.nodes())
    for edge in taxonomy_graph.edges:
        # Get adj matrix locations
        x = np.argwhere(nodes_array == edge[0])[0][0]
        y = np.argwhere(nodes_array == edge[1])[0][0]
        corr_val = correlation_matrix.values[x, y]
        if not np.isnan(corr_val):
            correlation_matrix_filtered[x, y] = corr_val
    if verbose:
        print("Total edges:")
        print(len(taxonomy_graph.edges))
        print("Measurable edges with data:")
        print(len(correlation_matrix_filtered[correlation_matrix_filtered > 0]))

    return correlation_matrix, correlation_matrix_filtered


def get_taxonomy_per_edge_correlation(
    taxonomy_graph: nx.classes.digraph.DiGraph,
    samples_dict: dict,
    verbose: bool = False,
) -> np.array:
    """
    Calculates the taxonomy edges correlation values using all possible data,
    this means that models are kept if they were tested on all datasets between
    two nodes that define an edge. The result is a correlation factor with
    potentially more models (data points) but the correlations of the different
    edges are calculated with different number of samples.
    The used sample dictionary is the unfiltered one.
    Returns a correlation matrix with values assigned only to valid edges.
    """

    # Calcualte the correlation matrix but using all the shared models between datasets edges
    correlation_matrix_imbalanced = np.zeros(
        (len(taxonomy_graph.nodes), len(taxonomy_graph.nodes))
    )
    nodes_array = np.array(taxonomy_graph.nodes())
    for edge in taxonomy_graph.edges:
        # Get adj matrix locations
        x = np.argwhere(nodes_array == edge[0])[0][0]
        y = np.argwhere(nodes_array == edge[1])[0][0]
        # get the models shared
        datasets_0 = taxonomy_graph.nodes[edge[0]].get("datasets", [])
        datasets_1 = taxonomy_graph.nodes[edge[1]].get("datasets", [])
        models_use_0 = list()
        models_use_1 = list()
        for dataset in datasets_0:
            values_dict = samples_dict.get(dataset, None)
            # Now fill matrix, data is already sorted
            if values_dict is not None:
                models_use_0 += list(values_dict.keys())
        for dataset in datasets_1:
            values_dict = samples_dict.get(dataset, None)
            # Now fill matrix, data is already sorted
            if values_dict is not None:
                models_use_1 += list(values_dict.keys())
        # Remove not shared
        element_counts = collections.Counter(models_use_0)
        models_use_0 = list()
        for key in element_counts.keys():
            if element_counts[key] == max(list(element_counts.values())):
                models_use_0.append(key)
        element_counts = collections.Counter(models_use_1)
        models_use_1 = list()
        for key in element_counts.keys():
            if element_counts[key] == max(list(element_counts.values())):
                models_use_1.append(key)
        # keep shared between nodes
        models_here_use = list()
        for model in models_use_0:
            if model in models_use_1:
                models_here_use.append(model)
        # Now build the metrics for each node in this edge
        values_0 = np.zeros(len(models_here_use))
        values_1 = np.zeros(len(models_here_use))
        for idx, model in enumerate(models_here_use):
            count = 0
            for dataset in datasets_0:
                values_dict = samples_dict.get(dataset, None)
                if values_dict is not None:
                    values_0[idx] += values_dict[model]
                    count += 1
            if count != 0:
                values_0[idx] /= count
            count = 0
            for dataset in datasets_1:
                values_dict = samples_dict.get(dataset, None)
                if values_dict is not None:
                    values_1[idx] += values_dict[model]
                    count += 1
            if count != 0:
                values_1[idx] /= count
        if np.sum(values_0) != 0 and np.sum(values_1) != 0:
            correlation_matrix_imbalanced[x, y] = np.corrcoef(values_0, values_1)[1, 0]

    return correlation_matrix_imbalanced
