import os
import yaml

def parse_yaml(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        dict_yaml = yaml.load(f, Loader=yaml.FullLoader)
        return dict_yaml