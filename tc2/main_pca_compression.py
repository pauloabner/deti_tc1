import numpy as np
from utils.dataset import load_breast_cancer_data
from models.lqm import LinearLeastSquaresClassifier
from models.pl_eq import PerceptronLogisticoEQ
from models.pl_ec import PerceptronLogisticoEC
from models.mlp import PerceptronMultiLayer as MLP
from utils.metrics import compute_statistics, accuracy_score
from utils.preprocessing import zscore_train_test, apply_pca_train_test

def run_experiments():
    ptrn = 0.8
    Nr = 50   

    filepath = "data/wdbc.data"
    X, d = load_breast_cancer_data(filepath)
    N_total, num_features = X.shape
    N_train = int(N_total * ptrn)

    components_list = [2, 5, 10, 15] # Apos a execução , escolher o melhor resultado

    models = {
        "LMQ": lambda: LinearLeastSquaresClassifier(),
        "PL/EQ": lambda: PerceptronLogisticoEQ(eta=0.01, epochs=100),
        "PL/EC": lambda: PerceptronLogisticoEC(eta=0.01, epochs=100),
        "MLP-1H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10,)),
        "MLP-2H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10, 5))
    }

    print(f"Dataset carregado: {num_features} atributos originais.")

    for k in components_list:
        print(f"\n{'='*70}")
        print(f" AVALIANDO DIMENSÃO q = {k} COMPONENTES PRINCIPAIS")
        print(f"{'='*70}")
        
        # Variável para acumular a variância média de todas as rodadas
        variances = []

        for model_name, model_constructor in models.items():
            accuracies = []
            execution_times = []
            
            for r in range(Nr):
                idx = np.random.permutation(N_total)
                train_idx = idx[:N_train]
                test_idx = idx[N_train:]

                X_train, X_test = X[train_idx], X[test_idx]
                d_train, d_test = d[train_idx], d[test_idx]

                X_train_pca, X_test_pca, var_ret = apply_pca_train_test(X_train, X_test, num_components=k)
                if model_name == "LMQ": # Coleta variância apenas uma vez por rodada
                    variances.append(var_ret)

                X_train_norm, X_test_norm = zscore_train_test(X_train_pca, X_test_pca)

                model = model_constructor()       
                train_time = model.fit(X_train_norm, d_train)

                d_pred = model.predict(X_test_norm)
                accuracy = accuracy_score(d_test, d_pred)
                
                accuracies.append(accuracy)
                execution_times.append(train_time)
            
            stats = compute_statistics(accuracies, execution_times)
            
            if model_name == "LMQ":
                print(f" -> Variância Média Preservada (Treino): {np.mean(variances):.2f}%")
                print("-" * 50)
                
            print(f" -> RESULTADOS: {model_name:<7} | q = {k}")
            print(f"  Média   : {stats['media']:.2f}%")
            print(f"  Mínimo  : {stats['minimo']:.2f}%")
            print(f"  Máximo  : {stats['maximo']:.2f}%")
            print(f"  Mediana : {stats['mediana']:.2f}%")
            print(f"  Desvio  : {stats['desvio_padrao']:.2f}")
            print(f"  Tempo   : {stats['tempo_total']:.4f}s\n")

if __name__ == "__main__":
    run_experiments()