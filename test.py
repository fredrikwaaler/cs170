import TSP as tsp

drive_percent = []
walk_percent = []
total_energy_cost = []

for i in range(10):
    dict = (tsp.algorithm('inputs/2_50.in', 'outputs/2_50.out'))
    drive = (dict["pathdistance"]/(dict["totalsum"]+dict["pathdistance"]))
    walk = (dict["totalsum"]/(dict["totalsum"]+dict["pathdistance"]))
    drive_percent.append(drive)
    walk_percent.append(walk)
    total_energy_cost.append((2/3)*dict["pathdistance"] + dict["totalsum"])




print("drive percent: ", drive_percent)
print("walk_percent: ", walk_percent)
print("total energy: ", total_energy_cost)



