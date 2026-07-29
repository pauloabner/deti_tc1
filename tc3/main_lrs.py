import numpy as np
import matplotlib.pyplot as plt
from utils import abner_math as am
from utils.lrs import local_random_search
from utils.dataset import load_gauss_data
from utils.abner_math import predict_polynomial, calculate_r2
if __name__ == "__main__":
    data = load_gauss_data()
    if data.size > 0:
        print(f"Carregadas {data.shape[0]} observações.")        
        
        y = data[:, 0]
        x = data[:, 1]

        # Normalização Min-Max de X (evita explosão numérica em polinômios de alto grau)
        x_min, x_max = np.min(x), np.max(x)
        x = (x - x_min) / (x_max - x_min)
        
        # --- PARÂMETROS DO LRS ---
        POLYNOMIAL_DEGREE = 10 # Melhor grau descoberto no TC1
        DIMENSION = POLYNOMIAL_DEGREE + 1 # a0 até a10
        NUM_RUNS = 500
        MAX_ITERATIONS = 5000
        
        
        LOWER_BOUND = -1000.0 
        UPPER_BOUND = 1000.0
        STANDARD_DEVIATION = 5.0
        
        results = []
        
        print(f"\nIniciando {NUM_RUNS} rodadas do LRS (Grau {POLYNOMIAL_DEGREE})...")
        for run in range(NUM_RUNS):
            best_beta, lowest_sse, history = local_random_search(
                x, y, LOWER_BOUND, UPPER_BOUND, MAX_ITERATIONS, DIMENSION, STANDARD_DEVIATION
            )
            
            results.append({
                'run_id': run + 1,
                'sse': lowest_sse,
                'beta': best_beta,
                'history': history
            })
            print(f"Rodada {run+1:02d} concluída | SEQ: {lowest_sse:.4e}")
            
        # Ordena para encontrar as 3 melhores (com base no erro SSE)
        results.sort(key=lambda d: d['sse'])
        top_3_results = results[:3]
        best_run = top_3_results[0]
        
        # Cálculos para o melhor modelo encontrado
        y_pred_best = predict_polynomial(x, best_run['beta'])
        r2_best = calculate_r2(y, y_pred_best)
        
        print(f"\nMelhor ajuste LRS encontrado na Rodada {best_run['run_id']}")
        print(f"SEQ: {best_run['sse']:.4e} | R²: {r2_best:.6f}")
        
        # ======================================================================
        # GRÁFICO 1: Convergência das 3 melhores rodadas
        # ======================================================================
        plt.figure(figsize=(10, 5))
        for res in top_3_results:
            plt.plot(res['history'], label=f"Rodada {res['run_id']} (SEQ={res['sse']:.2e})", linewidth=1.5)
        plt.title('Curvas de Convergência - LRS (As 3 Melhores Rodadas)')
        plt.xlabel('Iterações')
        plt.ylabel('Soma dos Erros Quadráticos (SEQ)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # ======================================================================
        # GRÁFICO 2: Dados Experimentais vs Melhor Curva LRS
        # ======================================================================
        plt.figure(figsize=(12, 7))
        plt.scatter(x, y, color='red', s=10, alpha=0.6, label='Dados Experimentais')
        plt.plot(x, y_pred_best, color='black', linewidth=2, label=f'Melhor LRS: Grau {POLYNOMIAL_DEGREE} (R²={r2_best:.3f})', zorder=5)
        plt.title('Ajuste Polinomial via Local Random Search (LRS) no Dataset Gauss3')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # ======================================================================
        # GRÁFICOS 3 e 4: Histograma de Resíduos e Dispersão
        # ======================================================================
        residuals = y - y_pred_best
        mu = np.mean(residuals)
        sigma = np.std(residuals)
        
        # Histograma
        plt.figure(figsize=(10, 5))
        plt.hist(residuals, bins=30, density=True, alpha=0.6, color='royalblue', edgecolor='black', label='Resíduos LRS')
        
        x_pdf = np.linspace(np.min(residuals), np.max(residuals), 100)
        gaussian_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - mu) / sigma)**2)
        plt.plot(x_pdf, gaussian_pdf, 'r-', linewidth=2, label=f'Gaussiana Analítica\n($\\mu$={mu:.3f}, $\\sigma$={sigma:.3f})')
        
        plt.title(f'Histograma dos Resíduos - LRS (Polinomial Grau {POLYNOMIAL_DEGREE})')
        plt.xlabel('Resíduo (Erro de Predição)')
        plt.ylabel('Densidade')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Dispersão
        plt.figure(figsize=(8, 8))
        plt.scatter(y, y_pred_best, color='darkorange', alpha=0.5, s=20, label='Real vs Modelo (LRS)')
        
        limite_min = min(np.min(y), np.min(y_pred_best))
        limite_max = max(np.max(y), np.max(y_pred_best))
        plt.plot([limite_min, limite_max], [limite_min, limite_max], 'k--', linewidth=2, label='Linha Ideal (Medido = Predito)')
        
        plt.title(f'Dispersão: Valor Medido vs. Valor Predito - LRS (Grau {POLYNOMIAL_DEGREE})')
        plt.xlabel('Valor Medido (Real)')
        plt.ylabel('Valor Predito (Modelo LRS)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.show()