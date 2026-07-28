import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from utils.dataset import load_breast_cancer_data
from models.lqm import LinearLeastSquaresClassifier
from models.pl_eq import PerceptronLogisticoEQ
from models.pl_ec import PerceptronLogisticoEC
from models.mlp import PerceptronMultiLayer as MLP
from utils.metrics import compute_statistics, accuracy_score
from utils.preprocessing import zscore_train_test

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_autoencoder_embeddings(X_train, X_test, q_dim):
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(q_dim,), 
        activation='logistic', # Ativação não-linear (sigmoide)
        solver='adam',
        max_iter=500,
        random_state=None
    )
    
    autoencoder.fit(X_train, X_train)
    
    W1 = autoencoder.coefs_[0]
    b1 = autoencoder.intercepts_[0]
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    X_train_emb = sigmoid(np.dot(X_train, W1) + b1)
    X_test_emb = sigmoid(np.dot(X_test, W1) + b1)
    
    return X_train_emb, X_test_emb

def run_experiments():
    ptrn = 0.8
    Nr = 50   

    filepath = "data/wdbc.data"
    X, d = load_breast_cancer_data(filepath)
    N_total, num_features = X.shape
    N_train = int(N_total * ptrn)
    
    # Mesma dimensão escolhida na Atividade 3
    q = 15 

    models = {
        "LMQ": lambda: LinearLeastSquaresClassifier(),
        "PL/EQ": lambda: PerceptronLogisticoEQ(eta=0.01, epochs=100),
        "PL/EC": lambda: PerceptronLogisticoEC(eta=0.01, epochs=100),
        "MLP-1H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10,)),
        "MLP-2H": lambda: MLP(eta=0.1, epochs=100, hidden_layer_sizes=(10, 5))
    }

    print(f"Dataset carregado: {num_features} atributos originais.")
    print(f"\n{'='*70}")
    print(f" AVALIANDO EMBEDDINGS NEURAIS (AUTOENCODER) COM q = {q}")
    print(f"{'='*70}")

    for model_name, model_constructor in models.items():
        accuracies = []
        execution_times = []
        
        for r in range(Nr):
            idx = np.random.permutation(N_total)
            train_idx = idx[:N_train]
            test_idx = idx[N_train:]

            X_train, X_test = X[train_idx], X[test_idx]
            d_train, d_test = d[train_idx], d[test_idx]

            X_train_norm, X_test_norm = zscore_train_test(X_train, X_test)
            
            X_train_emb, X_test_emb = get_autoencoder_embeddings(X_train_norm, X_test_norm, q_dim=q)

            X_train_emb_norm, X_test_emb_norm = zscore_train_test(X_train_emb, X_test_emb)

            model = model_constructor()       
            train_time = model.fit(X_train_emb_norm, d_train)

            d_pred = model.predict(X_test_emb_norm)
            accuracy = accuracy_score(d_test, d_pred)
            
            accuracies.append(accuracy)
            execution_times.append(train_time)
        
        stats = compute_statistics(accuracies, execution_times)
            
        print(f" -> RESULTADOS: {model_name:<7} | Autoencoder q={q}")
        print(f"  Média   : {stats['media']:.2f}%")
        print(f"  Mínimo  : {stats['minimo']:.2f}%")
        print(f"  Máximo  : {stats['maximo']:.2f}%")
        print(f"  Mediana : {stats['mediana']:.2f}%")
        print(f"  Desvio  : {stats['desvio_padrao']:.2f}")
        print(f"  Tempo   : {stats['tempo_total']:.4f}s\n")

if __name__ == "__main__":
    run_experiments()