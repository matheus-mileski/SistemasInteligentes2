from math import exp
import numpy as np


class MLP(object):
    """
    Uma rede neural(Perceptron-multilayer) de 3 camadas
    """

    def __init__(
        self,
        entrada,
        oculta,
        saida,
        taxaDeAprendizado=0.01,
        bias=1,
        pesosOculta=None,
        pesosSaida=None,
    ):
        """
        Entrada : número de entradas, de neurônios ocultos e saidas. Podendo também variar a taxa de aprendizado
        """
        self.entrada = entrada
        self.oculta = oculta
        self.saida = saida
        self.taxaDeAprendizado = taxaDeAprendizado
        self.bias = bias
        self.score = 0

        self.biasOculta = np.zeros((self.oculta, bias))
        self.biasSaida = np.zeros((self.saida, bias))

        np.random.seed(49)

        if pesosOculta is None:
            self.pesosOculta = np.random.uniform(-1, 1, (self.oculta, self.entrada))

        if pesosSaida is None:
            self.pesosSaida = np.random.uniform(-1, 1, (self.saida, self.oculta))

    def setScore(self, score):
        if score > self.score:
            self.score = score

    def getTaxaDeAprendizado(self):
        return self.taxaDeAprendizado

    def setTaxaDeAprendizado(self, taxa):
        self.taxaDeAprendizado = taxa

    def getParametros(self):
        params = {
            "neuroniosEntrada": self.entrada,
            "neuroniosOculta": self.oculta,
            "neuroniosSaida": self.saida,
            "taxaDeAprendizado": self.taxaDeAprendizado,
            "bias": self.bias,
            "pesosOculta": self.pesosOculta,
            "pesosSaida": self.pesosSaida,
            "score": self.score,
        }

        return params

    def ativacaoSigmoidal(self, valor):
        """
        Função ativadora Sigmoidal = 1 / (1 + e ^ - valor)
        Entrada : Valor a ser aplicado na função
        Retorno : Resultado da aplicação
        """
        exp_val = np.exp(-valor)
        return 1.0 / (1.0 + exp_val)

    def derivadaAtivacaoSigmoidal(self, valor):
        """
        Derivada da função ativadora Sigmoidal , dSigmoidal / dValor = Sigmoidal *(1 - Sigmoidal)
        Entrada : Valor(Resultante da aplicação à sigmoidal) a ser aplicado na função
        Retorno : Resultado da aplicação
        """
        return valor * (1 - valor)

    def erroQuadraticoMedio(self, esperado, valor):
        """
        Calculo do erro
        Entrada : O target e o valor deduzido
        Retorno : Erro calculado dadas as entradas
        """
        error = esperado - valor
        return error

    def feedForward(self, dados):
        """
        Recebe as entradas e faz a classificação
        Entrada : As N entradas(float) definidas no __init__
        Retorno : Nenhum
        """
        # Calcula as ativações da camada oculta
        neuroniosOculta = np.dot(self.pesosOculta, dados) + self.biasOculta
        saidaOculta = np.vectorize(self.ativacaoSigmoidal)(neuroniosOculta)

        # Calcula as ativações da camada de saída
        neuroniosSaida = np.dot(self.pesosSaida, saidaOculta) + self.biasSaida
        resultado = np.vectorize(self.ativacaoSigmoidal)(neuroniosSaida)
        return saidaOculta, resultado

    def backPropagation(self, dados, esperado):
        """
        Pondera as classificações e faz as correções aos pesos
        Entrada : Targets(float)
        Retorno : Nenhum
        """
        saidaOculta, resultado = self.feedForward(dados)

        # Calcula o erro da camada de saida
        erroSaida = self.erroQuadraticoMedio(esperado, resultado)
        deltaSaida = erroSaida * np.vectorize(self.derivadaAtivacaoSigmoidal)(resultado)

        # Calcula o erro na camada oculta
        erroOculta = np.dot(self.pesosSaida.T, deltaSaida)
        deltaOculta = erroOculta * np.vectorize(self.derivadaAtivacaoSigmoidal)(
            saidaOculta
        )

        # Atualiza pesos e bias
        self.pesosSaida += self.taxaDeAprendizado * np.dot(deltaSaida, saidaOculta.T)
        self.biasSaida += self.taxaDeAprendizado * deltaSaida
        self.pesosOculta += self.taxaDeAprendizado * np.dot(deltaOculta, dados.T)
        self.biasOculta += self.taxaDeAprendizado * deltaOculta

    def treinamento(self, dados, alvo, epocas=1000):
        for _ in range(epocas):
            for i in range(len(dados)):
                entrada = dados[i].reshape(-1, 1)
                esperado = alvo[i].reshape(-1, 1)
                self.feedForward(entrada)
                self.backPropagation(entrada, esperado)


def avaliaMLP(mlp, X_test, y_test):
    acertos = 0
    for i in range(len(X_test)):
        entrada = X_test[i].reshape(-1, 1)
        esperado = y_test[i].reshape(-1, 1)
        _, predito = mlp.feedForward(entrada)
        classePredita = np.argmax(predito)
        classeEsperada = np.argmax(esperado)
        if classePredita == classeEsperada:
            acertos += 1
    acuracia = acertos / len(X_test)
    return acuracia
