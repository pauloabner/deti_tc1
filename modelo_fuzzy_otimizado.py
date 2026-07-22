from utils.dataset import load_gauss_data
from utils import abner_math as am
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = load_gauss_data()
    print(f"Carregadas {data.shape[0]} observações.")
    x = data[:, 1]
    y = data[:, 0]

    idx_sort = np.argsort(x)
    x = x[idx_sort]
    y = y[idx_sort]

    r2_prev = -float('inf')
    best_nmf = 2
    best_yhat = None
    best_centers = None
    Nmf_max = 20
    for Nmf in range(2, Nmf_max + 1):
        centers = np.linspace(np.min(x), np.max(x), Nmf)
        spread = (centers[1] - centers[0]) / np.sqrt(2.0)
        pout = np.interp(centers, x, y)
        mi = np.zeros((Nmf, len(x)))
        for i in range(Nmf):
            mi[i, :] = np.exp(-((x - centers[i])**2) / (2 * spread**2))
        
        numerador = np.sum(mi * pout[:, np.newaxis], axis=0)
        denominador = np.sum(mi, axis=0)
        yhat = numerador / denominador
        
        r2 = am.r_squared(y, yhat)
        print(f"Nmf: {Nmf:2d} | R²: {r2:.6f}")

        if r2 >= r2_prev:                   
            r2_prev = r2
            best_nmf = Nmf
            best_yhat = yhat.copy()
            best_centers = centers.copy()
    print(f"\nMelhor modelo encontrado com Nmf={best_nmf} (R²: {r2_prev:.6f})")
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=10, alpha=0.5, label='Dados Experimentais')
    plt.plot(x, best_yhat, color='black', linewidth=3, label=f'Modelo Mamdani (Nmf={best_nmf})')

    best_pout = np.interp(best_centers, x, y)
    plt.scatter(best_centers, best_pout, color='red', marker='X', s=100, label='Centros (Regras)', zorder=5)

    plt.title(f'Aproximação Fuzzy Mamdani - Dataset Gauss3')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)


    # =========================================================================
    # PARTE B: METODOLOGIA DE AVALIAÇÃO (Gráficos para o Relatório)
    # =========================================================================
    
    # 1. Cálculo dos resíduos (erros de predição)
    residuos = y - best_yhat
    
    # Parâmetros da curva Gaussiana analítica (Média e Desvio Padrão dos resíduos)
    mu = np.mean(residuos)
    sigma = np.std(residuos)
    
    # --- GRÁFICO 1: Histograma de Resíduos ---
    plt.figure(figsize=(10, 5))
    
    # Plota o histograma normalizado (density=True é obrigatório para comparar com a curva)
    plt.hist(residuos, bins=30, density=True, alpha=0.6, color='royalblue', edgecolor='black')
    
    # Cria o eixo X para desenhar a curva teórica perfeita
    x_pdf = np.linspace(np.min(residuos), np.max(residuos), 100)
    
    # Equação analítica da PDF da Distribuição Normal
    pdf_gaussiana = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - mu) / sigma)**2)
    
    plt.plot(x_pdf, pdf_gaussiana, 'r-', linewidth=2, label=f'Gaussiana Analítica\n($\mu$={mu:.3f}, $\sigma$={sigma:.3f})')
    
    plt.title('Parte B.1 - Histograma dos Resíduos (Modelo Mamdani)')
    plt.xlabel('Resíduo (Erro de Predição)')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    
    # --- GRÁFICO 2: Dispersão Medido vs. Predito ---
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y, best_yhat, color='darkorange', alpha=0.5, s=20, label='Dados (Real vs Modelo)')
    
    # Calcula os limites para desenhar a reta diagonal ideal de 45 graus (onde Y_real = Y_predito)
    limite_min = min(np.min(y), np.min(best_yhat))
    limite_max = max(np.max(y), np.max(best_yhat))
    
    plt.plot([limite_min, limite_max], [limite_min, limite_max], 'k--', linewidth=2, label='Linha Ideal (Medido = Predito)')
    
    plt.title('Parte B.2 - Dispersão: Valor Medido vs. Valor Predito')
    plt.xlabel('Valor Medido (Real)')
    plt.ylabel('Valor Predito (Estimativa do Modelo)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Exibe todos os gráficos (A curva do modelo, o Histograma e a Dispersão)
    plt.show()


    plt.show()
        