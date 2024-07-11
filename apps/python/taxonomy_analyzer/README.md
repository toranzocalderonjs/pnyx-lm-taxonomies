# Taxonomy Analyzer

This folder contains a series of scripts that are used to analyze taxonomies.

### Testing (`test_taxonomy.py`)

This script is used to analyze a given taxonomy. The inputs are:
- A taxonomy file in `digraph` format.
- A set of metrics from multiple LMs on multiple datasets.

Example execution:
```bash
python taxonomy_analyzer/test_taxonomy.py --taxonomy /bar/foo/taxonomy_001.txt --data /bar/foo/helm/benchmark_output/runs/v0.3.0 --output /bar/foo/outputs
```

Note that the argument `--data` can be repeated multiple times to look for results in multiple folders, like this
```bash
...  --data /bar/foo/helm/benchmark_output/runs/v0.3.0  --data /bar/foo/custom_datasets ... 
```
