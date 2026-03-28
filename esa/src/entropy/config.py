import os, yaml

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open("%s/config.yaml" % (src_path + "/src"), "r") as file:
    CFG = yaml.safe_load(file)
