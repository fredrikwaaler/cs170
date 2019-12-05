import os
from TSP import algorithm


if __name__ == "__main__":
    for file in os.listdir(os.getcwd() + '/inputs'):
        print(file)
        algorithm(os.getcwd() + '/inputs/' + file, os.getcwd() + '/outputs/' + file.replace('.in', '.out'))



