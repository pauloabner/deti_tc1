import numpy as np
import matplotlib.pyplot as plt
from utils import abner_math as am
from utils.dataset import load_data

if __name__ == "__main__":
    data = load_data("aerogerador.dat")
    print(f"Carregadas {data.shape[0]} observações.")
    x = data[:, 0]
    y = data[:, 1] 

    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]  

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b.', label='Dados do Aerogerador')
    plt.grid(True)
    plt.title('CURVA DE POTENCIA - AEROGERADOR')
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência')

    # Passo 2: Definir o número de funções de pertinência e pontos de suporte
    Nmf = 5
    centers = np.array([4, 6.5, 9.2, 11.5, 14])
    powerw = np.array([20, 134, 270, 480, 508])
    
    # Plotando os centros como círculos vermelhos
    plt.plot(centers, powerw, 'ro', markersize=10, linewidth=3, label='Pontos de Suporte (Centros)')
    plt.legend()

    spread = 1.35
    xx = np.arange(0.8 * np.min(x), 1.2 * np.max(x) + 0.1, 0.1)
    # print (f"xx: {xx}")

    mi = np.zeros((Nmf, len(xx)))
    for i in range(Nmf):
        mi[i, :] = np.exp(-((xx - centers[i]) ** 2) / (2 * spread ** 2))
    # print (f"mi: {mi}")

    K = 200
    colors = ['r-', 'g-', 'b-', 'k-', 'm-']

    for i in range(Nmf):
        plt.plot(xx, K * mi[i, :], colors[i], linewidth=5, alpha=0.7)
    
    plt.figure(figsize=(10, 4))
    markerline, stemlines, baseline = plt.stem(powerw, np.ones(Nmf), linefmt='k-', markerfmt='ko', basefmt=' ')
    plt.setp(stemlines, linewidth=3)
    plt.setp(markerline, markersize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('PERTINENCIAS SINGLETON (SAIDA)')
    plt.xlabel('Potência')
    plt.ylabel('Grau de Pertinência \u03bc(p)')
    plt.ylim(0, 1.2)

    numerador = np.sum(mi * powerw[:, np.newaxis], axis=0)
    denominador = np.sum(mi, axis=0)
    yhat = numerador / denominador
    plt.figure(1)
    plt.plot(xx, yhat, 'k-', linewidth=4, label='Modelo Fuzzy Mamdani (Predição)')
    plt.legend()

    # 1. Calcula a matriz de pertinência para os dados reais 'x'
    mi_data = np.zeros((Nmf, len(x)))
    for i in range(Nmf):
        mi_data[i, :] = np.exp(-((x - centers[i])**2) / (2 * spread**2))
    
    # 2. Defuzzificação para os dados reais
    numerador_data = np.sum(mi_data * powerw[:, np.newaxis], axis=0)
    denominador_data = np.sum(mi_data, axis=0)
    yhat_data = numerador_data / denominador_data
    
    # 3. Cálculo do R2 usando sua biblioteca
    r2 = am.r_squared(y, yhat_data)
    print(f"R² do modelo Fuzzy Mamdani (tutorial): {r2:.4f}")
    plt.show()