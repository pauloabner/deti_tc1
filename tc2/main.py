import numpy as np
from utils.dataset import load_breast_cancer_data
from utils.preprocessing import zscore_train_test
from models.lqm import LinearLeastSquaresClassifier
from models.pl_eq import PerceptronLogisticoEQ
from utils.metrics import compute_statistics, accuracy_score

def run_experiment():
    ptrn = 0.8
    Nr = 50
    accuracies = []
    execution_times = []

    filepath = "data/wdbc.data"
    X, d = load_breast_cancer_data(filepath)
    N_total = X.shape[0]
    N_train = int(N_total * ptrn)

    for r in range(Nr):
        idx = np.random.permutation(N_total)
        train_idx = idx[:N_train]
        test_idx = idx[N_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        d_train, d_test = d[train_idx], d[test_idx]

        X_train_norm, X_test_norm = zscore_train_test(X_train, X_test)

        # model = LinearLeastSquaresClassifier()
        model = PerceptronLogisticoEQ(eta=0.1, epochs=150)
        train_time = model.fit(X_train_norm, d_train)

        d_pred = model.predict(X_test_norm)

        accuracy = accuracy_score(d_test, d_pred)
        accuracies.append(accuracy)

        execution_times.append(train_time)
    
    stats = compute_statistics(accuracies, execution_times)
    
    print("\n--- RESULTADOS FINAIS: CLASSIFICADOR LMQ ---")
    print(f"Média da Acurácia : {stats['media']:.2f}%")
    print(f"Desvio Padrão     : {stats['desvio_padrao']:.2f}")
    print(f"Mínimo            : {stats['minimo']:.2f}%")
    print(f"Máximo            : {stats['maximo']:.2f}%")
    print(f"Mediana           : {stats['mediana']:.2f}%")
    print(f"Tempo Total       : {stats['tempo_total']:.4f} segundos")


if __name__ == "__main__":
    run_experiment()