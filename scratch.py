import os
import sys
import json
config = {}


def reload_configuration():
    global config
    with open("settings.json", "r") as fp:
        config = json.load(fp)


reload_configuration()
print(config)

# def change_data(key, value):
#     with open("settings.json", "r") as fp:
#         json_data = json.load(fp)
#         print(json_data)
#     with open("settings.json", "w") as fp:
#         json_data['config'][key] = value
#         json.dump(json_data, fp, ensure_ascii=False)


# key = str(input("key : \n>"))
# value = str(input("value :\n>"))
# change_data(key, value)
