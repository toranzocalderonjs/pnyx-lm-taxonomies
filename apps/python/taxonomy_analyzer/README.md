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

### Dataset Compilation (`compile_custom_dataset.py`)

This script is used to compile a series of instance IDs from several dataset tests (from HELM) into a new "run" that has several compilations. The resulting dataset is saved in a format compatible with the HELM results format and can be used for the taxonomy testing script.

To use this script first we need to create a description of the compilation, for example:
```json
{
    "compilation_name" : "my_new_compilation",
    "datasets" : 
    [
        {
            "babi_qa" : 
            {
                "dataset_spec" : "task=all",
                "metric" : "quasi_exact_match",
                "split" : "test",
                "samples" : 
                {
                    "id5189" : "1",
                    "id37535" : "2",
                    "id37513" : "3",
                }
            }
        },
        {
            "other_dataset" : 
            {
                "dataset_spec" : "",
                "metric" : "quasi_exact_match",
                "split" : "test",
                "samples" : 
                {
                    "id111" : "2",
                    "id222" : "1",
                    "id333" : "4",
                }
            }
        }
    ]
}      
```
This `json` file contains the instruction to create a new run called `my_new_compilation` that will have 4 different compilations, called `c1`, `c2`, `c3`, `c4`. The run name and the compilation names can be any string. In this example the compilation `c1` and `c2` will contain samples from both datasets, while the compilation `c3` and `c4` from a single one. Keep in mind that if you wish to mix datasets in the compilations you must set the exact same `metric` and `split` for both.

To create the compilation just execute:
```bash
python taxonomy_analyzer/compile_custom_dataset.py --compilation /bar/foo/compilation_spec.json --data /bar/foo/helm/benchmark_output/runs/v0.4.0-instances --output /bar/foo/outputs/test_compilations
```

The `data` folder should contain all the instances data, this means, the `per_instance_stats.json` files.

The result of this execution will be a series of folders in the provided output (`/bar/foo/outputs/test_compilations`) with names following the HELM pattern, for example: `my_new_compilation:compilation=c1,model=meta_llama_30b`. Each folder will contain a `stats.json` file with the expected format:
```json
[
    {
        "name": {
            "name": "quasi_exact_match",
            "split": "test"
        },
        "count": 120,
        "sum": 81.0,
        "sum_squared": 81.0,
        "min": 0.0,
        "max": 1.0,
        "mean": 0.675,
        "variance": 0.21937499999999993,
        "stddev": 0.4703386943415967
    }
]
```

Finally in order to allow the `test_taxonomy.py` to read this new compilation, an entry should be added to the `helm_data.py` file (we need to make this easier yes...). For example something like this:

```python
    "my_new_compilation---compilation=c1": [
        {
            "name": "my_new_compilation:compilation=c1",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
```