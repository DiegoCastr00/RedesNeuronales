import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import datetime
###################################### DATA
# Cargar los datos desde el archivo CSV
data = pd.read_csv("HOG_skimage.csv")

# Dividir los datos en características (features) y etiquetas (labels)
X = data.drop("etiqueta", axis=1)  # Características
y = data["etiqueta"]  # Etiquetas

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

###################################### HYPERPARAMETERS
# We create a list with the parameters to be evaluated
hyperparameters = []
for random in np.arange(1, 100):
    hyperparameters.append([random])

###################################### EVALUATION
def evaluate_set(hyperparameter_set, results, lock, i, datas, N_HL):
    print("\n Yo soy el proceso:", i, "Comence a las:", datetime.datetime.now())

    # Hyperparameter to use
    for s in hyperparameter_set:

        HL = []
        for i in range (N_HL):
            HL.append(1)

        clf = MLPClassifier(
            max_iter = 200,
            solver= 'adam',
            tol = 0.000000001,

            # Hyperparameters
            learning_rate_init =0.20,
            momentum = 0.40,
            activation = "relu",
            hidden_layer_sizes = (38,),
            random_state=(int(s[0]))
        )

        scores = cross_validate(clf, X_train_scaled, y, cv=13, return_train_score=True)
        media_prueba = np.mean(scores['test_score'])
        media_entrenamiento = np.mean(scores['train_score'])
        with lock:
            print(f"=> media prueba: ", media_prueba)
            print(f"=> media entrenamiento: ", media_entrenamiento)
            print("--------------------------")
            # print("Puntuaciones: ", scores)
            datas.append([0.20, 0.40, N_HL,38,s[0],media_entrenamiento,media_prueba])
            # results.append(np.mean(scores))


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

    # Wait for all processes to finishsssss
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
    N_HL = 1
   # Crear un DataFrame de pandas con los datos
    df = pd.DataFrame(np.array(main(datas, N_HL)), columns=['Learning_rate', 'Momentum', 'N_capasOcultas','N_neuronas','Random State', 'Train_Accuracy', 'Test_Accuracy'])

    nombre_archivo = str(N_HL) + '1LayersNeuornas100.xlsx'
    df.to_excel(nombre_archivo, index=False)

    print("Archivo Excel guardado correctamente.")