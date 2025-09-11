
hSearcher = HyperparameterSearcher()
#hSearcher.grid_search(X, Y, [2, 3, 2, 1], h_range=(0.001, 0.05, 31), it_range=(20, 400, 41), nb_tests=100)
#hSearcher.grid_search(X, Y, [2, 3, 2, 1], h_range=(0.001, 0.05, 31), it_range=(400, 600, 21), nb_tests=100)
#hSearcher.grid_search(X, Y, [2, 3, 2, 1], h_range=(0.04, 0.05, 6), it_range=(550, 600, 6), nb_tests=200)
#hSearcher.grid_search(X, Y, [2, 3, 2, 1], h_range=(0.05, 0.1, 31), it_range=(20, 400, 41), nb_tests=100)

#print(hSearcher.grid_search(X, Y, [2, 3, 2, 1], h_range=(0.001, 0.1, 61), it_range=(100, 800, 71), nb_tests=200))
hSearcher.grid_search(X, Y, [2, 3, 2, 1], h_range=(0.04, 0.05, 6), it_range=(550, 600, 6), nb_tests=10)


class NeuralLayer:
    def __init__(self, nb_neurons: int, nb_ant: int):
        self.z = zeros((nb_neurons, 1)) # on code pour un seul état, mais les calculs s'adapteront pour
        self.a = zeros((nb_neurons, 1)) # calculer d'un seul bloc pour des données variées
        self.nb_neurons = nb_neurons
        self.nb_ants = nb_ant
        if nb_ant is not 0:
            c_lecun = sqrt(3/nb_ant)
            self.weights = random.uniform(-c_lecun, c_lecun, (nb_neurons, nb_ant))
            self.biases = zeros((nb_neurons, 1))
        else :
            self.biases = None
            self.weights = None

    @staticmethod
    def sigmoid(z: ndarray) -> ndarray:
        return where(z >= 0,
                        1/(1+exp(-z)),
                        exp(z)/(1+exp(z)))

    def forward(self, inputs: ndarray) -> ndarray:
        if inputs.shape[0] != self.weights.shape[1]:
            raise ValueError(f"Dimension incompatible: attendu {self.weights.shape[1]}, reçu {inputs.shape[0]}")
        self.z = self.weights @ inputs + self.biases
        self.a = self.sigmoid(self.z)
        return self.a


"""Pour 4 ou 5 cercles, réseau xx un peu juste, problème commun à xx, xxx, xxxx : si trop petit pas, complexification
Inutile : heuristique : foncer vers le résultat pour l'approcher sans s'arrêter au superficiel. -- > + de neurones.
xxx les meilleurs, pas 0.001, en 8-6-6 ou 8-8-4

Finesse du pas donne certains cas limites, plus de symétries et d'angles."""

"""
    @staticmethod
    def create_performance_heatmap(results: np.ndarray,
                                   h_range: Tuple[float, float, int],
                                   it_range: Tuple[int, int, int],
                                   title: str = "Carte de performance",
                                   interpolation: bool = False):

        Args :
            results : Matrice des résultats de performance
            h_range : (min, max, nb_points) pour learning rate
            it_range : (min, max, nb_points) pour les itérations
            title : Titre du graphique
            interpolation : si affichage d'une interpolation

        h_min, h_max, nb_h = h_range
        it_min, it_max, nb_it = it_range

        extent = [h_min, h_max, it_min, it_max]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Image brute
        im1 = ax1.imshow(results, vmin=0, vmax=1, extent=extent,
                         aspect='auto', origin="lower")
        ax1.set_title('Résultats bruts')
        ax1.set_xlabel('Taux d\'apprentissage')
        ax1.set_ylabel('Nombre d\'itérations')

        # Image interpolée
        im2 = ax2.imshow(results, extent=extent, vmin=0, vmax=1,
                         aspect='auto', interpolation='spline16', origin="lower")
        ax2.set_title('Résultats interpolés')
        ax2.set_xlabel('Taux d\'apprentissage')
        ax2.set_ylabel('Nombre d\'itérations')

        # Barre de couleur commune
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.subplots_adjust(right=0.8)
        fig.colorbar(im1, cax=cbar_ax, label='Taux de réussite')

        plt.suptitle(title)
        plt.show()
"""


#Complexifie la figure
"""X_bg = array([[X[k][0]-1, X[k][1]-1] for k in range(X.shape[0])])
X_bd = array([[X[k][0]+1, X[k][1]-1] for k in range(X.shape[0])])
X_hd = array([[X[k][0]+1, X[k][1]+1] for k in range(X.shape[0])])
X_hg = array([[X[k][0]-1, X[k][1]+1] for k in range(X.shape[0])])
X_p = array([[X[k][0] * 0.6, X[k][1] * 0.6] for k in range(X.shape[0])])

X_2, y_2 = concatenate((X_bg, X_bd, X_hg, X_hd), axis=0), concatenate((Y, Y, Y, Y), axis=0)
X_3, y_3 = concatenate((X_bg, X_bd, X_hg, X_hd, X_p), axis=0), concatenate((Y, Y, Y, Y, Y), axis=0)"""




#exemple_utilisation_complete()

# Création du réseau
cfg = NetworkConfig(
dimensions=[2, 12, 8, 1],
h=0.001,
it=6000,
r=0.02,
end_train=True,
affichage=True  )
#n = NetworkFactory.create_from_config(cfg)
# Visualisation avancée
#visualizer = NetworkVisualizer()


# Entraînement avec le nouveau "trainer"
#trainer = NetworkTrainer(n)
#history = trainer.train(X, Y, cfg)

# Affichage de la frontière de décision seulement
#visualizer.plot_decision_boundary(n, X, Y)

# Historique d'entraînement détaillé
#visualizer.plot_training_history(history)

# Évaluation de performance
#success_rate = trainer.evaluate_performance(X, Y, cfg, nb_tests=50, verbose=True)
#print(f"Taux de réussite: {success_rate:.2%}")


"""
    def searchRadiusSolutions(self, n_iter, n_max):
        r_init = []
        r_sol = []
        test = []
        for k in tqdm(range(n_iter)):
            network.construire(self.architecture)
            r_init.append(sqrt(self.carres()))
            self.entrainement(X, Y, 0.07, n_max, 0., affichage=False, end_train=True)
            test.append(self.end_train(X, Y))
            r_sol.append(sqrt(self.carres()))
        plt.scatter(r_init, r_sol, c=test)
        plt.show()
"""



def graph(zArray, x, y, p, lev, nbx, nby, acc, dim, extent):
    plt.imshow(zArray, extent=extent, vmin=0, vmax=1, aspect='auto', interpolation='bicubic', origin="lower")
    plt.colorbar()

    plt.suptitle('Avec un rapport coeff moindres carrés/pas de ' + str(p))
    plt.xlabel("Pas")
    plt.ylabel("Nombre d'itérations")
    plt.savefig("Carte_"+str(dim)+"_nbx_" + str(nbx) + "_nby_" + str(nby) + "_acc_" + str(acc) + '_Imshow.png', transparent=True, dpi=150)
    plt.show()
    modelisation(x, extent, zArray)

    plt.contourf(x, y, zArray, lev)
    plt.colorbar()
    plt.suptitle('Avec un rapport coeff moindres carrés/pas de ' + str(p))
    plt.xlabel("Pas")
    plt.ylabel("Nombre d'itérations")
    plt.savefig(
        "Carte_"+str(dim)+"_nbx_" + str(nbx) + "_nby_" + str(nby) + "_acc_" + str(acc) + '_contourf.png', transparent=True, dpi=150)
    plt.show()
    modelisation(x, extent, zArray)

def modelisation(x, extent, zarray):
    hyp = [2/(p+0.001) for p in x]
    plt.plot(x, hyp)
    plt.imshow(zarray, extent=extent, vmin=0, vmax=1, aspect='auto', interpolation='bicubic', origin="lower")
    plt.colorbar()
    plt.xlabel("Pas")
    plt.ylabel("Nombre d'itérations")
    plt.show()




def train_fast(n: Network,
               data: ndarray,
               model: ndarray,
               config: 'NetworkConfig',
               verbose: bool = False) -> int:
    """
    Entraînement rapide avec early stopping.
    Retourne l'itération finale atteinte.
    """
    h, it, r = config.lrng_rate, config.it, config.r
    rp, m = 2 * r, model.shape[0]
    layers, p = n.layers, len(n.layers) - 1
    end_train = config.end_train

    iteration_range = tqdm(range(it), desc="Training") if verbose else range(it)

    for k in iteration_range:
        # Forward (pas besoin de stocker la sortie)
        n.forward_propagation(data)
        dz = layers[-1].a - model

        # Backward
        for j in range(p):
            x = p - j
            a_prev = layers[x - 1].a
            layer = layers[x]

            dW = dz @ a_prev.T / m
            db = dz.sum(axis=1, keepdims=True) / m

            if x != 1:
                dz = (layer.weights.T @ dz) * a_prev * (1 - a_prev)

            layer.weights -= h * (dW + rp * layer.weights)
            layer.biases -= h * (db + rp * layer.biases)

        # Vérification convergence
        if end_train and end_train_check(n, data, model):
            if verbose:
                print(f"Convergence atteinte à l'itération {k}")
            return k  # stop immédiat

    return it + 1  # pas convergé avant la fin


"""from numpy import random, where, ndarray, exp, zeros, sqrt

class NeuralLayer:
    def __init__(self, nb_neurons: int, nb_ant: int):
        self.z = zeros((nb_neurons, 1))        # on code pour un seul état, mais les calculs s'adapteront pour
        self.a = zeros((nb_neurons, 1)) # calculer d'un seul bloc pour des données variées
        self.nb_neurons = nb_neurons
        self.nb_ants = nb_ant
        if nb_ant != 0:
            c_lecun = sqrt(3.0 / self.nb_ants)
            rng = random.default_rng()
            self.biases = random.randn(nb_neurons, 1)  # un nombre réel selon la distribution normale
            self.weights = rng.uniform(-c_lecun, c_lecun, (self.nb_neurons, self.nb_ants)).astype(float)
            #self.weights = random.randn(nb_neurons, nb_ant)
        else :
            self.biases = None
            self.weights = None

    @staticmethod
    def sigmoid(z: ndarray) -> ndarray:
        return where(z >= 0, 1/(1+exp(-z)), exp(z)/(1+exp(z)))

    def forward(self, inputs: ndarray) -> ndarray:
        if inputs.shape[0] != self.weights.shape[1]:
            raise ValueError(f"Dimension incompatible: attendu {self.weights.shape[1]}, reçu {inputs.shape[0]}")
        self.z = self.weights @ inputs + self.biases
        self.a = self.sigmoid(self.z)
        return self.a

    def initialize(self):
        nb_ant, nb_neurons = self.nb_ants, self.nb_neurons
        if nb_ant != 0:
            c_lecun = sqrt(3/nb_ant)
            self.weights = random.uniform(-c_lecun, c_lecun, (nb_neurons, nb_ant))
            self.biases = zeros((nb_neurons, 1))

class Network:
    def __init__(self):
        self.layers: list[NeuralLayer] = []
        self.architecture: list[int] = []
        self.training_history = {'loss':[], 'accuracy':[]}

    def add_layer(self, nb_neurons: int) -> None:
        if not self.layers:
            layer = NeuralLayer(nb_neurons, 0)
        else:
            layer = NeuralLayer(nb_neurons, self.layers[-1].nb_neurons)
        self.layers.append(layer)
        self.architecture.append(nb_neurons)

    def build_network(self, architecture: list[int]) -> None:
        self.layers.clear()
        self.architecture.clear()
        for nb_neurons in architecture:
            self.add_layer(nb_neurons)

    def forward_propagation(self, data: ndarray) -> ndarray:
        if data.shape[0] != self.architecture[0]:
            raise ValueError(f"Dimension d'entrée incorrecte: {data.shape[0]} vs {self.architecture[0]}")
        self.layers[0].a = data
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1].a)
        return self.layers[-1].a"""


@staticmethod
def plot_decision_boundary(n: Network,
                           data: ndarray,
                           model: ndarray,
                           cadre_x: Tuple[float, float, int] = (-2, 2, 41),
                           cadre_y: Tuple[float, float, int] = (-2, 2, 41),
                           title: str = "Frontière de décision"):
    x_min, x_max, nb_x = cadre_x
    y_min, y_max, nb_y = cadre_y
    x = linspace(x_min, x_max, nb_x)
    y = linspace(y_min, y_max, nb_y)

    xx, yy = meshgrid(x, y)
    zz = array([n.forward_propagation(array([xx[i], yy[i]]))[0]
                for i in range(len(y))])

    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, zz, levels=50, alpha=0.8)
    plt.colorbar(label='Probabilité de classe 1')
    plt.scatter(data[0, :], data[1, :], c=model[0, :], cmap='RdYlBu', s=100, edgecolors='black')
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    plt.show()
