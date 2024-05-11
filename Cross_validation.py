import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import datetime
###################################### DATA
# Cargar los datos desde el archivo CSV
data = pd.read_csv("HOG_bb.csv")

# Dividir los datos en características (features) y etiquetas (labels)
X = data.drop("etiqueta", axis=1)  # Características
y = data["etiqueta"]  # Etiquetas

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)



###################################### HYPERPARAMETERS
# We create a list with the parameters to be evaluated
hyperparameters = []
for learning_rate in np.arange(0.2, 0.5, 0.1):
    for momentum_descent in np.arange(0.2, 0.5, 0.1):
        for random in np.arange(2, 20):
            hyperparameters.append([learning_rate, momentum_descent, random])

###################################### EVALUATION
def evaluate_set(hyperparameter_set, results, lock, i, datas, N_HL):
    print("\n Yo soy el proceso:", i, "Comence a las:", datetime.datetime.now())

    # Hyperparameter to use
    for s in hyperparameter_set:

        HL = []
        for i in range (N_HL):
            HL.append(int(s[2]))

        clf = MLPClassifier(
            max_iter = 1000,
            solver= 'adam',
            validation_fraction = 0.1,
            tol = 1e-4,

            # Hyperparameters
            learning_rate_init = float(s[0]),
            momentum = float(s[1]),
            activation = "logistic",
            random_state= int(s[2]),
            hidden_layer_sizes = (11,11),
        )
        # Realiza la validación cruzada en el conjunto de entrenamiento
        scores = cross_val_score(clf, X_train_scaled, y, cv=10)
        
        with lock:
            print(f"=> media: ", np.mean(scores))
            # print("Puntuaciones: ", scores)
            datas.append([s[0], s[1], N_HL, 11, s[2],np.mean(scores)])
            results.append(np.mean(scores))


######################################## MAIN
def main(datas, N_HL):

    # Initialize shared data structures
    manager = multiprocessing.Manager()
    results = manager.list()

    # Now we will evaluate with multiple processes
    processes = []
    N_PROCESSES = 4
    splits = np.array_split(hyperparameters, N_PROCESSES)
    lock = multiprocessing.Lock()

    start_time = time.perf_counter()

    for i in range(N_PROCESSES):
        # Generate the processing threads
        p = multiprocessing.Process(
            target=evaluate_set, args=(splits[i], results, lock, i, datas, N_HL)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Time
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")

    # Process the results
    print("Results:")
    for i, acc in enumerate(results):
        print(f"Accuracy for process {i}: {acc}")

    return datas


######################################## WRITE XLSX
if __name__ == "__main__":

    manager = multiprocessing.Manager()
    datas = manager.list()

    # Hidden layers
    N_HL = 2

    # Crear un DataFrame de pandas con los datos
    df = pd.DataFrame(np.array(main(datas, N_HL)), columns=['Learning_rate', 'Momentum', 'N_capasOcultas','N_neuronas','Random State', 'Precision'])

    nombre_archivo = str(N_HL) + 'Prueba2Layers.xlsx'
    df.to_excel(nombre_archivo, index=False)

    print("Archivo Excel guardado correctamente.")