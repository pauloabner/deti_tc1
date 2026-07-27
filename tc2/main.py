import numpy as np
from utils.dataset import load_breast_cancer_data
from models.lqm import LinearLeastSquaresClassifier
from models.pl_eq import PerceptronLogisticoEQ
from models.pl_ec import PerceptronLogisticoEC
from models.mlp import PerceptronMultiLayer as MLP
from utils.metrics import compute_statistics, accuracy_score
from utils.preprocessing import (
    no_normalization, 
    zscore_train_test, 
    minmax_01_normalize, 
    minmax_bipolar_normalize
)

def run_experiment():
    ptrn = 0.8
    Nr = 50   

    filepath = "data/wdbc.data"
    X, d = load_breast_cancer_data(filepath)
    N_total = X.shape[0]
    N_train = int(N_total * ptrn)

    models = {
        "LMQ": lambda: LinearLeastSquaresClassifier(),
        "PL/EQ": lambda: PerceptronLogisticoEQ(eta=0.01, epochs=100),
        "PL/EC": lambda: PerceptronLogisticoEC(eta=0.01, epochs=100),
        "MLP-1H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10,)),
        "MLP-2H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10, 5))
    }

    normalizations = {
        "Sem Normalizacao": no_normalization,
        "Z-Score": zscore_train_test,
        "Min-Max [0, +1]": minmax_01_normalize,
        "Min-Max [-1, +1]": minmax_bipolar_normalize
    }

    for model_name, model_constructor in models.items():
        print(f"\n--- Executando Experimento para o Classificador: {model_name} ---")
        best_acc = 0.0
        best_norm = ""
        best_stats = None
        for norm_name, norm_function in normalizations.items():
            accuracies = []
            execution_times = []
            for r in range(Nr):
                idx = np.random.permutation(N_total)
                train_idx = idx[:N_train]
                test_idx = idx[N_train:]

                X_train, X_test = X[train_idx], X[test_idx]
                d_train, d_test = d[train_idx], d[test_idx]

                X_train_norm, X_test_norm = norm_function(X_train, X_test)
  
                model = model_constructor()       
                train_time = model.fit(X_train_norm, d_train)

                d_pred = model.predict(X_test_norm)

                accuracy = accuracy_score(d_test, d_pred)
                accuracies.append(accuracy)

                execution_times.append(train_time)
            
            stats = compute_statistics(accuracies, execution_times)
            # print(f" -> Acurácia Média: {stats['media']:.2f}% (±{stats['desvio_padrao']:.2f})")
                
            # Verifica se é a melhor configuração até agora
            if stats['media'] > best_acc:
                best_acc = stats['media']
                best_norm = norm_name
                best_stats = stats
        
            print(f"\n -> RESULTADOS FINAIS: CLASSIFICADOR {model_name} - {norm_name} ---")
            print(f"  Média da Acurácia : {stats['media']:.2f}%")
            print(f"  Desvio Padrão     : {stats['desvio_padrao']:.2f}")
            print(f"  Mínimo            : {stats['minimo']:.2f}%")
            print(f"  Máximo            : {stats['maximo']:.2f}%")
            print(f"  Mediana           : {stats['mediana']:.2f}%")
            print(f"  Tempo Total       : {stats['tempo_total']:.4f} segundos")
            print(f" -> Normalização: {norm_name:<20} | Acurácia Média: {stats['media']:.2f}% (±{stats['desvio_padrao']:.2f})")
    
    print(f" MELHOR CONFIGURAÇÃO PARA {model_name}: {best_norm}")
    print(f"Valores para a Tabela 1:")
    print(f"Média  : {best_stats['media']:.2f}%")
    print(f"Desvio : {best_stats['desvio_padrao']:.2f}")
    print(f"Mínimo : {best_stats['minimo']:.2f}%")
    print(f"Máximo : {best_stats['maximo']:.2f}%")
    print(f"Mediana: {best_stats['mediana']:.2f}%")
    print(f"Tempo  : {best_stats['tempo_total']:.4f}s")
    print("-" * 50)


if __name__ == "__main__":
    run_experiment()