from math import exp
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


class MLP(object):
	"""
	Uma rede neural(Perceptron-multilayer) de 3 camadas
	"""
	def __init__(self,entrada, oculta, saida, taxaDeAprendizado = 0.1, bias = 1, weights_input_hidden = None, weights_hidden_output = None):
		"""
		Entrada : número de entradas, de neurônios ocultos e saidas. Podendo também variar a taxa de aprendizado
		"""
		self.entrada = entrada
		self.oculta = oculta
		self.saida = saida
		self.taxaDeAprendizado = taxaDeAprendizado

		self.bias_hidden = np.zeros((self.oculta, bias))
		self.bias_output = np.zeros((self.saida, bias))

		if weights_input_hidden is None:
			self.weights_input_hidden = np.random.uniform(-1, 1, (self.oculta, self.entrada))
		
		if weights_hidden_output is None:
			self.weights_hidden_output = np.random.uniform(-1, 1, (self.saida, self.oculta))
		
	
	def getTaxaDeAprendizado(self):
		return self.taxaDeAprendizado
		
	def setTaxaDeAprendizado(self,taxa):
		self.taxaDeAprendizado = taxaDeAprendizado

	def ativacaoSigmoidal(self, valor):
		"""
		Função ativadora Sigmoidal = 1 / (1 + e ^ - valor)
		Entrada : Valor a ser aplicado na função
		Retorno : Resultado da aplicação
		"""
		if valor >= 0:
			return 1.0 / (1.0 + np.exp(-valor))
		else:
			exp_val = np.exp(valor)
			return exp_val / (1.0 + exp_val)

	def derivadaAtivacaoSigmoidal(self, valor):
		"""
		Derivada da função ativadora Sigmoidal , dSigmoidal / dValor = Sigmoidal *(1 - Sigmoidal)
		Entrada : Valor(Resultante da aplicação à sigmoidal) a ser aplicado na função
		Retorno : Resultado da aplicação
		"""
		sigmoidal = self.ativacaoSigmoidal(valor)
		return sigmoidal * (1 - sigmoidal)

	def erroQuadraticoMedio(self, esperado, valor):
		"""
		Calculo do erro
		Entrada : O target e o valor deduzido
		Retorno : Erro calculado dadas as entradas
		"""		
		error = esperado - valor
		return np.sum(error**2)
		
	def feedForward(self, dados):
		"""
		Recebe as entradas e faz a classificação
		Entrada : As N entradas(float) definidas no __init__
		Retorno : Nenhum
		"""
		# Calculate the activations of the hidden layer
		hidden_inputs = np.dot(self.weights_input_hidden, dados) + self.bias_hidden
		hidden_outputs = np.vectorize(self.ativacaoSigmoidal)(hidden_inputs)

        # Calculate the activations of the output layer
		output_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
		output_outputs = np.vectorize(self.ativacaoSigmoidal)(output_inputs)

		return hidden_outputs, output_outputs

	def backPropagation(self, dados, esperado):
		"""
		Pondera as classificações e faz as correções aos pesos
		Entrada : Targets(float)
		Retorno : Nenhum
		"""
		hidden_outputs, output_outputs = self.feedForward(dados)

        # Calculate the error in the output layer
		output_errors = esperado - output_outputs
		#output_delta = output_errors * self.derivadaAtivacaoSigmoidal(output_outputs)
		output_delta = output_errors * np.vectorize(self.ativacaoSigmoidal)(output_outputs)

        # Calculate the error in the hidden layer
		hidden_errors = np.dot(self.weights_hidden_output.T, output_delta)
		#hidden_delta = hidden_errors * self.derivadaAtivacaoSigmoidal(hidden_outputs)
		hidden_delta = hidden_errors * np.vectorize(self.ativacaoSigmoidal)(hidden_outputs)

        # Update weights and biases
		self.weights_hidden_output += self.taxaDeAprendizado * np.dot(output_delta, hidden_outputs.T)
		self.bias_output += self.taxaDeAprendizado * output_delta
		self.weights_input_hidden += self.taxaDeAprendizado * np.dot(hidden_delta, dados.T)
		self.bias_hidden += self.taxaDeAprendizado * hidden_delta

	
	def treinamento(self, dados, esperado, epochs=1000):
		for _ in range(epochs):
			for i in range(len(dados)):
				input_data = dados[i].reshape(-1, 1)
				target_data = esperado[i].reshape(-1, 1)
				self.feedForward(input_data)
				self.backPropagation(input_data, target_data)
		

def evaluate_mlp(mlp, X_test, y_test):
	correct = 0
	for i in range(len(X_test)):
		input_data = X_test[i].reshape(-1, 1)
		target_data = y_test[i].reshape(-1, 1)
		_, output = mlp.feedForward(input_data)
		predicted_class = np.argmax(output)
		true_class = np.argmax(target_data)
		if predicted_class == true_class:
			correct += 1
	accuracy = correct / len(X_test)
	return accuracy

if __name__ == "__main__":
	# Testing with the iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target

	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	y_train = to_categorical(y_train, num_classes=3)
	y_test = to_categorical(y_test, num_classes=3)

	entrada = 4  # Number of input features in the Iris dataset
	#oculta = 15   # Number of hidden neurons (adjust as needed)
	saida = 3    # Number of output classes (Iris has 3 classes)

	for oculta in range(1,20):
		mlp = MLP(entrada, oculta, saida, taxaDeAprendizado=0.1)
		mlp.treinamento(X_train, y_train, epochs=1000)
		print(f"{oculta} layers")
		accuracy = evaluate_mlp(mlp, X_test, y_test)
		print(f"Test Accuracy: {accuracy * 100:.2f}%")