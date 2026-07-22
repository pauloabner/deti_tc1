import numpy as np
import matplotlib.pyplot as plt
from utils import abner_math as am
from utils.dataset import load_gauss_data

if __name__ == "__main__":
    data = load_gauss_data()
    if data.size > 0:
        print(f"Carregadas {data.shape[0]} observações.")
        
        # Extraindo as colunas: y é a primeira (0), x é a segunda (1)
        y = data[:, 0]
        x = data[:, 1]

        plt.figure(figsize=(12, 7))
        plt.scatter(x, y, color='red', s=10, alpha=0.6, label='Dados Experimentais')

        dim = 0
        prev_r2 = -float('inf')
        best_g = None
        
        print("Iniciando busca iterativa por melhor grau...")
        while dim < 20: # Limite para evitar sobreajuste excessivo ou erro de memória
            try:
                g = am.Gauss(y, x, dim)
                
                # Se o R² parou de melhorar, interrompemos
                if g.r2 <= prev_r2:
                    break
                
                print(f"Grau {dim}: R² = {g.r2:.6f}")
                plt.plot(x, g.y_pred, label=f'Grau {dim} (R²={g.r2:.3f})', alpha=0.4, linewidth=1)
                
                prev_r2 = g.r2
                best_g = g
                dim += 1
            except np.linalg.LinAlgError:
                print(f"Matri´z singular atingida no grau {dim}. Parando.")
                break

        # Destaca a melhor curva encontrada
        if best_g:
            print(f"\nMelhor ajuste encontrado no Grau {best_g.dim} (R²: {best_g.r2:.6f})")
            plt.plot(x, best_g.y_pred, color='black', linewidth=2, label=f'MELHOR: Grau {best_g.dim}', zorder=5)
            print(f"Beta final: {best_g.beta}")
            # Cálculo da correlação entre y (real) e y_pred (modelo)
            correlacao = np.corrcoef(y, best_g.y_pred)[0, 1]
            print(f"Correlação (r): {correlacao:.6f}")
            print(f"Correlação ao quadrado (r²): {correlacao**2:.6f} (Deve ser igual ao R²)")

        plt.title('Evolução do Ajuste Polinomial no Dataset Gauss3')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)

        if best_g:
            # Cálculo dos resíduos do melhor modelo
            residuos = y - best_g.y_pred
            
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
            
            plt.title(f'Parte B - Histograma dos Resíduos (Polinomial Grau {best_g.dim})')
            plt.xlabel('Resíduo (Erro de Predição)')
            plt.ylabel('Densidade')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # --- GRÁFICO 2: Dispersão Medido vs. Predito ---
            plt.figure(figsize=(8, 8))
            plt.scatter(y, best_g.y_pred, color='darkorange', alpha=0.5, s=20, label='Real vs Modelo')
            
            # Linha ideal de 45 graus
            limite_min = min(np.min(y), np.min(best_g.y_pred))
            limite_max = max(np.max(y), np.max(best_g.y_pred))
            plt.plot([limite_min, limite_max], [limite_min, limite_max], 'k--', linewidth=2, label='Linha Ideal (Medido = Predito)')
            
            plt.title(f'Parte B - Dispersão: Valor Medido vs. Valor Predito (Grau {best_g.dim})')
            plt.xlabel('Valor Medido (Real)')
            plt.ylabel('Valor Predito (Modelo)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()