import os

if __name__ == "main":
    for file in os.listdir("{}/inputs".format(os.curdir)):
        print(file)
