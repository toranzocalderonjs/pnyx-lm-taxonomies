import json

#Read the input JSON file
with open('./config/helm_tests.json', 'r') as file:
    data =json.load(file)

extracted_data = []

#Extract data name
for list_item in data:
    print(list_item)
    dataset_name = list_item
    extracted_data.append(dataset_name)

#Create JSON list of names of datasets in helm_tests.json
with open('./outputs/datasets/dataset_list.json', 'w') as file:
    json.dump(extracted_data, file, indent=4)
