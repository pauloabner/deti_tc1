import numpy as np
from utils.abner_math import predict_polynomial, calculate_sse, calculate_sae

def particle_swarm_optimization(x_data, y_data, num_particles, dimensions, bounds, max_iterations, c1=2.0, c2=2.0, w=0.7, error_metric=calculate_sse):
    """
    Implementação do algoritmo PSO (Particle Swarm Optimization) padrão para minimização de erro.
    
    Parâmetros:
    - x_data, y_data: Dados experimentais (X deve estar normalizado entre 0 e 1).
    - num_particles: Tamanho da população (número de partículas no enxame).
    - dimensions: Número de variáveis a otimizar (para grau 10, são 11 variáveis).
    - bounds: Tupla (lower_bound, upper_bound) indicando o espaço de busca.
    - max_iterations: Número máximo de iterações do algoritmo.
    - c1: Coeficiente cognitivo (peso da memória individual). Padrão comum: 2.0.
    - c2: Coeficiente social (peso do grupo). Padrão comum: 2.0.
    - w: Inércia (peso da velocidade anterior). Padrão comum: 0.7.
    """
    lower_bound, upper_bound = bounds
    
    # 1. Inicialização do Enxame
    # Posições (X) e Velocidades (V) aleatórias para cada partícula
    particles_position = np.random.uniform(lower_bound, upper_bound, (num_particles, dimensions))
    # Velocidade inicial limitada a uma fração do espaço de busca
    max_vel = (upper_bound - lower_bound) * 0.1 
    particles_velocity = np.random.uniform(-max_vel, max_vel, (num_particles, dimensions))
    
    # Avaliação inicial
    # pbest (personal best): Melhor posição que cada partícula já visitou
    pbest_position = np.copy(particles_position)
    pbest_error = np.zeros(num_particles)
    
    for i in range(num_particles):
        y_pred = predict_polynomial(x_data, particles_position[i])
        pbest_error[i] = error_metric(y_data, y_pred)
        
    # gbest (global best): Melhor posição encontrada por QUALQUER partícula do enxame
    gbest_index = np.argmin(pbest_error)
    gbest_position = np.copy(pbest_position[gbest_index])
    gbest_error = pbest_error[gbest_index]
    
    error_history = [gbest_error]
    
    # 2. Loop Principal (Iterações)
    for _ in range(max_iterations):
        for i in range(num_particles):
            # Vetores de números aleatórios r1 e r2 (entre 0 e 1)
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            
            # Atualização da Velocidade
            # V(t+1) = w*V(t) + c1*r1*(pbest - X(t)) + c2*r2*(gbest - X(t))
            cognitive_component = c1 * r1 * (pbest_position[i] - particles_position[i])
            social_component = c2 * r2 * (gbest_position - particles_position[i])
            
            particles_velocity[i] = (w * particles_velocity[i]) + cognitive_component + social_component
            
            # Atualização da Posição
            # X(t+1) = X(t) + V(t+1)
            particles_position[i] = particles_position[i] + particles_velocity[i]
            
            # (Opcional, mas recomendado) Aplicar limites (Clamping) para evitar que a partícula fuja do espaço de busca
            particles_position[i] = np.clip(particles_position[i], lower_bound, upper_bound)
            
            # Avaliação da nova posição
            y_pred_cand = predict_polynomial(x_data, particles_position[i])
            current_error = error_metric(y_data, y_pred_cand)
            
            # Atualização do pbest (Melhor Individual)
            if current_error < pbest_error[i]:
                pbest_position[i] = np.copy(particles_position[i])
                pbest_error[i] = current_error
                
                # Atualização do gbest (Melhor Global)
                if current_error < gbest_error:
                    gbest_position = np.copy(particles_position[i])
                    gbest_error = current_error
                    
        # Gravar histórico do melhor erro global nesta iteração
        error_history.append(gbest_error)
        
    return gbest_position, gbest_error, error_history