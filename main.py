from neuron_class import Network
from network_config_factory import NetworkConfig

from numpy import array, linspace, zeros, ndarray
from typing import Tuple
from tqdm import tqdm

from network_trainer import train_fast, train, evaluate_performance, end_train_check
from network_visualizer import create_performance_heatmap, display_training_results, display_search_results

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=100, noise=0.1, factor=0.2, random_state=0)
Y = array([[y[k]] for k in range(len(y))])
X, Y = X.T, Y.T

default_config = NetworkConfig(dimensions=[2, 3, 2, 1],
                               lrng_rate=0.05,
                               it=1000,
                               r=0.02,  #regularization coefficient
                               end_train=True)

# Exemple d'utilisation intégrée avec votre workflow existant
def exemple_of_use():
    # Création du réseau
    n_ex = Network()
    n_ex.build_network(default_config.dimensions)
    history_dict = train(n_ex, X, Y, default_config)

    # Affichage de la frontière de décision seulement
    display_training_results(n_ex, X, Y, default_config, history_dict, saved=True)

    # Évaluation de performance
    success_rate_ex = evaluate_performance(X, Y, n_ex, default_config.it, default_config.lrng_rate, default_config.r, nb_tests=50)
    print(f"For 50 similarly trained MLPs, perfect reproduction in {success_rate_ex:.2%} of cases.")


def grid_search(data: ndarray,
                model: ndarray,
                architecture: list[int],
                learning_rate_range: Tuple[float, float, int] = (0.01, 0.1, 20),
                it_range: Tuple[int, int, int] = (100, 1000, 10),
                regularization: float = 0.02,
                nb_tests: int = 20,
                interpolation: bool = False,
                print_result_array: bool = False) -> ndarray:
    """
        Studies on a grid the efficiency of configurations by varying the learning rate and the number of iterations.
    """
    h_min, h_max, nb_h = learning_rate_range
    it_min, it_max, nb_it = it_range

    h_values = linspace(h_min, h_max, nb_h)
    it_values = linspace(it_min, it_max, nb_it, dtype=int)

    nw = Network()
    nw.build_network(architecture)
    results = zeros((nb_it, nb_h))

    for i, lrng_rate in enumerate(tqdm(h_values, desc="progress of mapping")):
        for j, it in enumerate(it_values):
            results[j, i] = evaluate_performance(data, model, nw, it, lrng_rate, regularization, nb_tests)
    if print_result_array:
        for l in results:
            print(l)

    # Visualisation automatique
    create_performance_heatmap(
        results, learning_rate_range, it_range,
        f"Performance - Architecture {architecture}",
        interpolation=interpolation
    )
    return results

def search_radius_solutions(nb: int,
                            data: ndarray,
                            model: ndarray,
                            config: NetworkConfig = default_config):
    r_init, r_sol, r_b_sol, r_w_sol, result = [], [], [], [], []

    n = Network()
    n.build_network(config.dimensions)
    it, lrng_rate, r, end_train = config.it, config.lrng_rate, config.r, config.end_train

    for _ in tqdm(range(nb)):
        Network.re_initialize(n.layers)
        r_init.append(n.compute_norm()[0])
        train_fast(n, data, model, it, lrng_rate, r, end_train)

        result.append(end_train_check(n, data, model))
        t = n.compute_norm()
        r_sol.append(t[0])
        r_w_sol.append(t[1])
        r_b_sol.append(t[2])
    display_search_results(array(r_init), array(r_sol), array(r_w_sol), array(r_b_sol), array(result), config)


search_config = NetworkConfig(dimensions=[2, 3, 3, 1],
                               lrng_rate=0.07,
                               it=1000,
                               r=0.01,  #regularization coefficient
                               end_train=True)
search_radius_solutions(1200, X, Y, search_config)