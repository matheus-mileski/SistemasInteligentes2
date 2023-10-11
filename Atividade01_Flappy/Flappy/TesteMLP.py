from MLP import MLP, avaliaMLP
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.utils import to_categorical

def teste():
	# Testando com o dataset Iris
	iris = load_iris()
	X = iris.data
	y = iris.target

	# Separa o dataset em treinamento e teste
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	y_train = to_categorical(y_train, num_classes=3)
	y_test = to_categorical(y_test, num_classes=3)

	entrada = len(iris.feature_names)  # Numero de features de entrada do dataset Iris
	#oculta = 15   # Numero de neuronios na camada oculta
	saida = len(iris.target_names)    # Numero de classes do dataset Iris

	for oculta in range(1,20):
		mlp = MLP(entrada, oculta, saida, taxaDeAprendizado=0.1)
		mlp.treinamento(X_train, y_train, epocas=1000)
		print(f"{oculta} neuronios na camada oculta")
		acuracia = avaliaMLP(mlp, X_test, y_test)
		print(f"Acurácia: {acuracia * 100:.2f}%")

if __name__ == "__main__":
    teste()

