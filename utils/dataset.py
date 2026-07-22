import numpy as np
def load_gauss_data(filepath="Gauss3.dat"):
    """
    Reads Gauss3.dat and returns a NumPy array of the data.
    """
    data = []
    try:
        with open(filepath, 'r') as file:

            lines = file.readlines()
            
            # Find the index where 'data' occurs (followed by X and Y values)
            start_index = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('Data:   y          x'):
                    start_index = i + 1
                    break
            
            if start_index != -1:
                # Extract and parse the remaining lines
                for line in lines[start_index:]:
                    parts = line.split()
                    if parts:
                        data.append([float(p) for p in parts])
        return np.array(data)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return np.array([])

def load_data(filepath="data.dat"):
    """
    Reads data.dat and returns a NumPy array of the data.
    """
    data = []
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.split()
                if parts:
                    data.append([float(p) for p in parts])
        return np.array(data)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return np.array([])