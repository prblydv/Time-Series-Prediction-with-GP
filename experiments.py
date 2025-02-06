import sys
import numpy as np
import math
from main import question1


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # Generate data points according to the equation x + y = z
    x = np.random.uniform(-100, 100, 500)
    y = np.random.uniform(-100, 100, 500)
    z = np.random.uniform(-100, 100, 500)
    a = np.random.uniform(-100, 100, 500)

    res = (x*y) + (z*a)

    # Create a DataFrame
    # data = pd.DataFrame({'x': x, 'y': y, 'z': z, 'a': a, 'res': res})
    data = pd.DataFrame({'x': x, 'y' : y, 'z': z, 'a': a, 'res': res})


    # Save the data to a tab-separated file
    file_path = 'addition_2'
    data.to_csv(file_path, sep='\t', index=False, header=False)

    file_path






