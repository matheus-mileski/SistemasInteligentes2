# Criar venv e instalar requirements
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```
# Testar o MLP com o Iris

```
python MLP.py
```

# Ideias

- Possíveis entradas do MLP:
    - Posição Vertical do FlappyBird
    - Velocidade Vertical do FlappyBird
    - Distância pro próximo Pipe
    - Diferença de altura entre o FlappyBird e a parte inferior do Pipe
    - Diferença de altura entre a parte inferior e superior do Pipe
    - Score

- Camadas de entrada diferentes para cada agente do AG:
    - Salva uma lista com todos os parametros para cada FlappyBird.
    - Na criação da primeira população sorteia para cada FlappyBird uma quantidade de 1 - 6 de parâmetros (neurônios de entrada) e depois sorteia os índices de 0 - 5.


- Sorteia um número de 1 - 100 para a quantidade de neurônios na camada oculta.

- Camada de saída fixa com 1 neurônio.

- Função de Fitness: Score (?)

- Seleção de indivíduos: Roleta ou Torneio (tamanho do torneio) (?)

- Reprodução: (?)

- Parâmetros de cada agente: 
    - nº de neurônios de entrada
    - indices dos neurônios de entrada
    - nº de neurônios da camada oculta
    - bias
    - pesos de entrada
    - pesos de saída