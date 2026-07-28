from models.lqm import LinearLeastSquaresClassifier
from models.pl_eq import PerceptronLogisticoEQ
from models.pl_ec import PerceptronLogisticoEC
from models.mlp import PerceptronMultiLayer as MLP
from utils.dataset import load_breast_cancer_data
from utils.metrics import compute_statistics, accuracy_score
from utils.preprocessing import zscore_train_test, apply_pca
import numpy as np

def run_pca_experiment():
    ptrn = 0.8
    Nr = 50 

    filepath = "data/wdbc.data"
    X, d = load_breast_cancer_data(filepath) 
    N_total, num_features_original = X.shape
    N_train = int(N_total * ptrn)

    # Componentes a serem testados
    components_list = [2, 5, 10, 15]

    models = {
        "LMQ": lambda: LinearLeastSquaresClassifier(),
        "PL/EQ": lambda: PerceptronLogisticoEQ(eta=0.01, epochs=100),
        "PL/EC": lambda: PerceptronLogisticoEC(eta=0.01, epochs=100),
        "MLP-1H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10,)),
        "MLP-2H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10, 5))
    }

    print(f"Dataset carregado: {num_features_original} atributos originais.")

    model_name = "LMQ"


    for model_name, model_constructor in models.items():
        print(f"\n{'='*60}")
        print(f"--- Classificador: {model_name} (com PCA) ---")
        print(f"{'='*60}")
        for k in components_list:
            accuracies = []
            execution_times = []
            variance_info = 0.0

            for r in range(Nr):
                idx = np.random.permutation(N_total)
                train_idx = idx[:N_train]
                test_idx = idx[N_train:]

                X_train, X_test = X[train_idx], X[test_idx]
                d_train, d_test = d[train_idx], d[test_idx]

                # Normalização (Z-Score)
                X_train_norm, X_test_norm = zscore_train_test(X_train, X_test)

                X_train_pca, X_test_pca, var_retained = apply_pca(X_train_norm, X_test_norm, num_components=k)
                variance_info = var_retained 

                model = model_constructor()       
                train_time = model.fit(X_train_pca, d_train)

                d_pred = model.predict(X_test_pca)
                accuracy = accuracy_score(d_test, d_pred)
                
                accuracies.append(accuracy)
                execution_times.append(train_time)
            
            stats = compute_statistics(accuracies, execution_times)

            print(f"\n -> Componentes: {k} (Variância Retida: {variance_info:.2f}%)")
            print(f"  Média da Acurácia : {stats['media']:.2f}% (±{stats['desvio_padrao']:.2f})")
            print(f"  Tempo Total       : {stats['tempo_total']:.4f} segundos")



if __name__ == "__main__":
    run_pca_experiment()