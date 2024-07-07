import json
from collections import OrderedDict

def json_conversion(json_file):

    with open(json_file, 'r') as file:
        data = json.load(file)

    unique_signs = list(OrderedDict((item['sign'], None) for item in data).keys())

    return unique_signs