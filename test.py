import pandas as pd
import numpy as np
import copy
import math
import time
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations
import random


def draw_bar_plot(filename, listx):
    df = pd.read_csv(filename)
    # Data for the bar plot

    # Set bar width and adjust positions for grouped bars
    width = 0.25
    listx1 = [x - (width / 2) for x in range(len(listx))]  # Shift left
    listx2 = [x + (width / 2) for x in range(len(listx))]  # Shift right

    # Y-axis data
    listy1 = df["GA"]
    listy2 = df["Optimal"]

    plt.bar(listx1, listy1, width, label="GA")
    plt.bar(listx2, listy2, width, label="Opt")

    # Set x-ticks and labels
    plt.xticks(range(len(listx)), labels=listx)

    # Add legend, title, and axis labels
    plt.legend()
    plt.title("Makespan Comparison on OR-Library Instances")
    plt.ylabel("Makesapn")
    plt.xlabel("15*10 OR Library Instances")

    # Show the plot
    plt.savefig("Makespan Comparison_on OR-Library Instances Bar Chart")
    plt.show()



draw_bar_plot("Result.csv", ["la"+str(i+1) for i in range(25, 30)])
