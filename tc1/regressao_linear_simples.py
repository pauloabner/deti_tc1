from utils import abner_math as am
from utils.dataset import load_gauss_data
import numpy as np


if __name__ == "__main__":
    data = load_gauss_data()
    if data.size > 0: 
        y = data[:, 0]
        x = data[:, 1]
        print(f"Carregadas {data.shape[0]} observações.")
        print(f"Comprimento de x: {len(x)}")
        print(f"Comprimento de y: {len(y)}")
    
    xbarra = np.mean(x)
    ybarra = np.mean(y)
    print(f"Valor de xbarra: {xbarra}")
    print(f"Valor de ybarra: {ybarra}")

    Sx2 = np.cov(x, ddof=1)
    print(f"Valor de Sx2: {Sx2}")

    Sxy = np.cov(x, y, ddof=1)[0, 1]
    print(f"Valor de Sxy: {Sxy}")

    b1 = Sxy / Sx2
    b0 = ybarra - b1 * xbarra
    print(f"Valor de b1: {b1}")
    print(f"Valor de b0: {b0}")