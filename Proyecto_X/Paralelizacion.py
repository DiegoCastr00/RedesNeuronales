import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import datetime

###################################### DATA
# Cargar los datos desde el archivo CSV
data = pd.read_csv("HOG.csv")

# Dividir los datos en características (features) y etiquetas (labels)
X = data.drop("etiqueta", axis=1)  # Características
y = data["etiqueta"]  # Etiquetas

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X) ## 100% test (cross validation)


###################################### HYPERPARAMETERS
# We create a list with the parameters to be evaluated
hyperparameters = []
for learning_rate in np.arange(0.2, 0.9, 0.2):
    for momentum_descent in np.arange(0.2, 0.9, 0.2):
        for hidden_layers in np.arange(3, 16):
            for r_state in np.arange(10, 16):
                hyperparameters.append([learning_rate, momentum_descent, hidden_layers, r_state])

###################################### EVALUATION
def evaluate_set(hyperparameter_set, results, lock, i, datas, N_HL):
    print("\n Yo soy el proceso:", i, "Comence a las:", datetime.datetime.now())

    # Hyperparameter to use
    for s in hyperparameter_set:

        HL = []
        for i in range (N_HL):
            HL.append(int(s[2]))

        mlp_classifier = MLPClassifier(
            max_iter = 1000,
            activation = 'logistic',
            solver= 'adam',
            tol = 1e-3,

            # Hyperparameters
            learning_rate_init = float(s[0]),
            momentum = float(s[1]),
            hidden_layer_sizes = HL,
            random_state= int(s[3])
        )

        ## 80/20
        # clf.fit(X_train_scaled, y_train)
        # y_pred = clf.predict(X_test_scaled)

        ## Cross val
        # Realiza la validación cruzada en el conjunto de entrenamiento
        scores = cross_validate(mlp_classifier, X_train, y, cv=15,  return_train_score=True)

        with lock:
            # print("Lr: ", s[0], " | M: ", s[1], " | N_HL: ", N_HL, " | N_Neu: ", s[2], " | R_state: ", s[3]," | Precision_train(prom): ", scores['train_score'].mean(), " | Precision_test(prom): ", scores['test_score'].mean())
            # datas.append([s[0], s[1], N_HL, s[2], s[3], accuracy_score(y_test,y_pred)]) ## 80/20
            datas.append([s[0], s[1], N_HL, s[2], s[3], scores['train_score'].mean(), scores['test_score'].mean()]) ## Cross val
            # results.append(accuracy_score(y_test, y_pred)) ## 80/20
            results.append(scores['test_score'].mean()) ## Cross val


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
    N_HL = 3

    # Crear un DataFrame de pandas con los datos
    df = pd.DataFrame(np.array(main(datas, N_HL)), columns=['LR', 'M', 'N_HL','N_Neu', 'R_state', 'Precision_train(prom)', 'Precision_test(prom)'])

    nombre_archivo = str(N_HL) + 'HL_HOG.xlsx'
    df.to_excel(nombre_archivo, index=False)

    print("Archivo Excel guardado correctamente.")