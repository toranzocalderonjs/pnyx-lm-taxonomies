# This structure contains the translation of the taxonomy dataset strings
# into actual HELM dataset results.
# An entry in this dictionary does not need to be exactly one dataset, nor be
# constituted of the same dataset, arbitrary collections can be passed. Only
# keep in mind that the metrics should all be of the same type, since all
# results within a dataset entry will be averaged.
dataset_config = {

#####################################
# ------ Singleton databases ------ #
    "LegalSupport": [
        {
            "name": "legal_support,method=multiple_choice_joint",  # Partial string of the result name, all the characters before ",model=..."
            "metric": "exact_match",  # Type of metric to look for in the "stats.json" file. The first match is the one used.
            "field": "mean",  # Field of the metric to keep, usually mean.
            "split": "test",  # Split of the metric, some dataset have "test" split, other only have "validation"
        }
    ],
    "bAbI": [
        {
            "name": "babi_qa:task=all",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "LSAT": [
        {
            "name": "lsat_qa:task=all",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "HellaSwag": [
        {
            "name": "commonsense:dataset=hellaswag",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        }
    ],
    "OpenBookQA": [
        {
            "name": "commonsense:dataset=openbookqa",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "WikiText-103": [
        {
            "name": "IGNORE-ME",
        }
    ],
    "WikiData": [
        {
            "name": "IGNORE-ME",
        }
    ],
    "dyck_language": [
        {
            "name": "dyck_language_np=3",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "entity_matching_Abt_Buy": [
        {
            "name": "entity_matching:dataset=Abt_Buy",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "entity_matching_Beer": [
        {
            "name": "entity_matching:dataset=Beer",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "entity_matching_Dirty_iTunes_Amazon": [
        {
            "name": "entity_matching:dataset=Dirty_iTunes_Amazon",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "natural_qa_closedbook": [
        {
            "name": "natural_qa:mode=closedbook",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        }
    ],
    "natural_qa_openbook_longans": [
        {
            "name": "natural_qa:mode=openbook_longans",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "truthful_qa": [
        {
            "name": "truthful_qa:task=mc_single,method=multiple_choice_joint",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "boolq": [
        {
            "name": "boolq",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "msmarco_regular": [
        {
            "name": "IGNORE-ME",
        }
    ],
    "msmarco_trec": [
        {
            "name": "IGNORE-ME",
        }
    ],
    "quac": [
        {
            "name": "quac",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],   
    "wikifact_author": [
        {
            "name": "wikifact:k=5,subject=author",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_currency": [
        {
            "name": "wikifact:k=5,subject=currency",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_discoverer_or_inventor": [
        {
            "name": "wikifact:k=5,subject=discoverer_or_inventor",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_instance_of": [
        {
            "name": "wikifact:k=5,subject=instance_of",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_medical_condition_treated": [
        {
            "name": "wikifact:k=5,subject=medical_condition_treated",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_part_of": [
        {
            "name": "wikifact:k=5,subject=part_of",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_place_of_birth": [
        {
            "name": "wikifact:k=5,subject=place_of_birth",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_plaintiff": [
        {
            "name": "wikifact:k=5,subject=plaintiff",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_position_held": [
        {
            "name": "wikifact:k=5,subject=position_held",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "wikifact_symptoms_and_signs": [
        {
            "name": "wikifact:k=5,subject=symptoms_and_signs",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "synthetic_reasoning_natural_easy": [
        {
            "name": "synthetic_reasoning_natural:difficulty=hard",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "synthetic_reasoning_natural_hard": [
        {
            "name": "synthetic_reasoning_natural:difficulty=hard",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "synthetic_reasoning_induction": [
        {
            "name": "synthetic_reasoning:mode=induction",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "synthetic_reasoning_pattern_match": [
        {
            "name": "synthetic_reasoning:mode=pattern_match",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "synthetic_reasoning_variable_substitution": [
        {
            "name": "synthetic_reasoning:mode=variable_substitution",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    
########################################
# ------ Non-splitted databases ------ #
    "Synthetic_reasoning_natural_language": [
        {
            "name": "synthetic_reasoning_natural:difficulty=hard",
            "metric": "f1_set_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "synthetic_reasoning_natural:difficulty=easy",
            "metric": "f1_set_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "Synthetic_reasoning_abstract_symbols": [
        {
            "name": "synthetic_reasoning:mode=pattern_match",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "synthetic_reasoning:mode=variable_substitution",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "synthetic_reasoning:mode=induction",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "MMLU": [
        {
            "name": "mmlu:subject=abstract_algebra",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=college_chemistry",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=computer_security",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=econometrics",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
        {
            "name": "mmlu:subject=us_foreign_policy",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
    ],
    "The_Pile": [
        {
            "name": "the_pile:subset=ArXiv",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=BookCorpus2",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=Enron Emails",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=Github",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=PubMed Central",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "the_pile:subset=Wikipedia (en)",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "TwitterAAE": [
        {
            "name": "twitter_aae:demographic=white",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "twitter_aae:demographic=aa",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "ICE": [
        {
            "name": "ice:gender=female",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:gender=male",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=ea",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=hk",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=ind",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "ice:subset=usa",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "BLiMP": [
        {
            "name": "blimp:phenomenon=binding,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "blimp:phenomenon=irregular_forms,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "blimp:phenomenon=island_effects,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
        {
            "name": "blimp:phenomenon=quantifiers,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field": "mean",
            "split": "test",
        },
    ],

####################################
# ------ Splitted databases ------ #
    'BLiMP---binding' : [
        {
            "name" : "blimp:phenomenon=binding,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field" : "mean",
            "split": "test",
        }
    ],
    'BLiMP---irregular_forms' : [
        {
            "name" : "blimp:phenomenon=irregular_forms,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field" : "mean",
            "split": "test",
        }
    ],
    'BLiMP---island_effects' : [
        {
            "name" : "blimp:phenomenon=island_effects,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field" : "mean",
            "split": "test",
        }
    ],
    'BLiMP---quantifiers' : [
        {
            "name" : "blimp:phenomenon=quantifiers,method=multiple_choice_separate_original",
            "metric": "exact_match",
            "field" : "mean",
            "split": "test",
        }
    ],
    "ICE---gender---female": [
        {
            "name": "ice:gender=female",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        }
    ],
    "ICE---gender---male": [
        {
            "name": "ice:gender=male",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        }
    ],
    "ICE---subset---ea": [
        {
            "name": "ice:subset=ea",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        }
    ],
    "ICE---subset---hk": [
        {
            "name": "ice:subset=hk",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        }
    ],
    "ICE---subset---ind": [
        {
            "name": "ice:subset=ind",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        }
    ],
    "ICE---subset---usa": [
        {
            "name": "ice:subset=usa",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "TwitterAAE---white": [
        {
            "name": "twitter_aae:demographic=white",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "TwitterAAE---aa": [
        {
            "name": "twitter_aae:demographic=aa",
            "metric": "bits_per_byte",
            "field": "mean",
            "split": "test",
        },
    ],
    "Synthetic_reasoning_abstract_symbols---pattern_match": [
        {
            "name": "synthetic_reasoning:mode=pattern_match",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "Synthetic_reasoning_abstract_symbols---variable_sustitution": [
        {
            "name": "synthetic_reasoning:mode=variable_substitution",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "Synthetic_reasoning_abstract_symbols---induction": [
        {
            "name": "synthetic_reasoning:mode=induction",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "Synthetic_reasoning_natural_language---hard": [
        {
            "name": "synthetic_reasoning_natural:difficulty=hard",
            "metric": "f1_set_match",
            "field": "mean",
            "split": "test",
        }
    ],
    "Synthetic_reasoning_natural_language---easy": [
        {
            "name": "synthetic_reasoning_natural:difficulty=easy",
            "metric": "f1_set_match",
            "field": "mean",
            "split": "test",
        }
    ],
    'The_Pile---ArXiv' : [
        {
            "name" : "the_pile:subset=ArXiv",
            "metric": "bits_per_byte",
            "field" : "mean",
            "split": "test",
        }
    ],
    'The_Pile---BookCorpus2' : [
        {
            "name" : "the_pile:subset=BookCorpus2",
            "metric": "bits_per_byte",
            "field" : "mean",
            "split": "test",
        }
    ],
    'The_Pile---Enron Emails' : [
        {
            "name" : "the_pile:subset=Enron Emails",
            "metric": "bits_per_byte",
            "field" : "mean",
            "split": "test",
        }
    ],
    'The_Pile---Github' : [
        {
            "name" : "the_pile:subset=Github",
            "metric": "bits_per_byte",
            "field" : "mean",
            "split": "test",
        }
    ],
    'The_Pile---PubMed Central' : [
        {
            "name" : "the_pile:subset=PubMed Central",
            "metric": "bits_per_byte",
            "field" : "mean",
            "split": "test",
        }
    ],
    'The_Pile---Wikipedia' : [
        {
            "name" : "the_pile:subset=Wikipedia (en)",
            "metric": "bits_per_byte",
            "field" : "mean",
            "split": "test",
        }
    ],
    "MMLU---abstract_algebra": [
        {
            "name": "mmlu:subject=abstract_algebra",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        }
    ],
    "MMLU---college_chemistry": [
        {
            "name": "mmlu:subject=college_chemistry",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        }
    ],
    "MMLU---computer_security": [
        {
            "name": "mmlu:subject=computer_security",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        }
    ],
    "MMLU---econometrics": [
        {
            "name": "mmlu:subject=econometrics",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        }
    ],
    "MMLU---us_foreign_policy": [
        {
            "name": "mmlu:subject=us_foreign_policy",
            "metric": "exact_match",
            "field": "mean",
            "split": "valid",
        },
    ],

    ########## BABI TASKS #############
    "babi_qa_Task_1": [
        {
            "name": "babi_qa:subtask_compilation=1",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_2": [
        {
            "name": "babi_qa:subtask_compilation=2",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_3": [
        {
            "name": "babi_qa:subtask_compilation=3",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_4": [
        {
            "name": "babi_qa:subtask_compilation=4",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_5": [
        {
            "name": "babi_qa:subtask_compilation=5",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_6": [
        {
            "name": "babi_qa:subtask_compilation=6",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_7": [
        {
            "name": "babi_qa:subtask_compilation=7",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_8": [
        {
            "name": "babi_qa:subtask_compilation=8",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_9": [
        {
            "name": "babi_qa:subtask_compilation=9",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_10": [
        {
            "name": "babi_qa:subtask_compilation=10",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_11": [
        {
            "name": "babi_qa:subtask_compilation=11",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_12": [
        {
            "name": "babi_qa:subtask_compilation=12",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_13": [
        {
            "name": "babi_qa:subtask_compilation=13",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_14": [
        {
            "name": "babi_qa:subtask_compilation=14",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_15": [
        {
            "name": "babi_qa:subtask_compilation=15",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_16": [
        {
            "name": "babi_qa:subtask_compilation=16",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_17": [
        {
            "name": "babi_qa:subtask_compilation=17",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_18": [
        {
            "name": "babi_qa:subtask_compilation=18",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_19": [
        {
            "name": "babi_qa:subtask_compilation=19",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],
    "babi_qa_Task_20": [
        {
            "name": "babi_qa:subtask_compilation=20",
            "metric": "quasi_exact_match",
            "field": "mean",
            "split": "test",
        },
    ],    
}
