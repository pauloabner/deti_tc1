from utils.dataset import load_gauss_data
from utils import abner_math as am
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    data = load_gauss_data()
    if data.size > 0:
        print(f"Carregadas {data.shape[0]} observações.")
        
        # Extraindo as colunas: y é a primeira (0), x é a segunda (1)
        y = data[:, 0]
        x = data[:, 1]

        plt.figure(figsize=(12, 7))
        plt.scatter(x, y, color='red', s=10, alpha=0.6, label='Dados Experimentais')

        num_particoes = 20
        r2_prev = -float('inf') # Começa bem baixo para a primeira comparação funcionar
        best_k = 1
        best_y_pred = None
        best_p = None
        for k in range(1, num_particoes + 1):
            p = np.linspace(np.min(x), np.max(x), k + 1)
            # print(f"Partições: {p}")
            y_pred = np.zeros_like(y)
            for i in range(k):
                if i == k - 1:
                    mask = (x >= p[i]) & (x <= p[i + 1])
                else:
                    mask = (x >= p[i]) & (x < p[i + 1])
                x_block = x[mask]
                y_block = y[mask]

                X_block = am.matrix_X(1, x_block)
                beta_block = am.beta_estimate(X_block, y_block)
                y_pred[mask] = am.y_predict(X_block, beta_block)
            r2 = am.r_squared(y, y_pred)
            print(f"R² para {k} partições: {r2:.6f}")  
            if r2 >= r2_prev:   
                r2_prev = r2
                best_k = k
                best_y_pred = y_pred.copy()
                best_p = p
        
        for i in range(best_k):
            if i == best_k - 1:
                mask = (x >= best_p[i]) & (x <= best_p[i + 1])
            else:
                mask = (x >= best_p[i]) & (x < best_p[i + 1])
            
            # Plota a reta da partição i com uma legenda única
            if i == 0:
                plt.plot(x[mask], best_y_pred[mask], color='black', linewidth=2, label=f'Melhor Ajuste ({best_k} partes)')
            else:
                plt.plot(x[mask], best_y_pred[mask], color='black', linewidth=2)
        
        # plt.plot(x_block, y_pred[mask], color='blue', linewidth=2, label=f'Intervalo {i+1}')

    plt.title(f'Modelo Linear por Partes no Dataset Gauss3 (Melhor k={best_k})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5) 

    # =========================================================================
    # PARTE B: METODOLOGIA DE AVALIAÇÃO (Gráficos para o Relatório)
    # =========================================================================
    
    # Cálculo dos resíduos do melhor modelo
    residuos = y - best_y_pred
    
    # Parâmetros da curva Gaussiana analítica (Média e Desvio Padrão)
    mu = np.mean(residuos)
    sigma = np.std(residuos)
    
    # --- GRÁFICO 1: Histograma de Resíduos ---
    plt.figure(figsize=(10, 5))
    plt.hist(residuos, bins=30, density=True, alpha=0.6, color='royalblue', edgecolor='black', label='Resíduos')
    
    # Criando a curva analítica
    x_pdf = np.linspace(np.min(residuos), np.max(residuos), 100)
    pdf_gaussiana = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - mu) / sigma)**2)
    plt.plot(x_pdf, pdf_gaussiana, 'r-', linewidth=2, label=f'Gaussiana Analítica\n($\\mu$={mu:.3f}, $\\sigma$={sigma:.3f})')
    
    plt.title('Parte B - Histograma dos Resíduos (Linear por Partes)')
    plt.xlabel('Resíduo (Erro de Predição)')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # --- GRÁFICO 2: Dispersão Medido vs. Predito ---
    plt.figure(figsize=(8, 8))
    plt.scatter(y, best_y_pred, color='darkorange', alpha=0.5, s=20, label='Real vs Modelo')
    
    # Linha ideal de 45 graus
    limite_min = min(np.min(y), np.min(best_y_pred))
    limite_max = max(np.max(y), np.max(best_y_pred))
    plt.plot([limite_min, limite_max], [limite_min, limite_max], 'k--', linewidth=2, label='Linha Ideal (Medido = Predito)')
    
    plt.title('Parte B - Dispersão: Valor Medido vs. Valor Predito')
    plt.xlabel('Valor Medido (Real)')
    plt.ylabel('Valor Predito (Modelo)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()