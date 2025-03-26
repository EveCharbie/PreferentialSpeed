import matplotlib.pyplot as plt
import numpy as np
from get_data import get_data
from graphs import plot_data

data_path = "/Volumes/Eve_Mac/Eve_data"

data = get_data(data_path)

plot_data(data)
