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

- Possнveis entradas do MLP:
    - Posiзгo Vertical do FlappyBird
    - Velocidade Vertical do FlappyBird
    - Distвncia pro prуximo Pipe
    - Diferenзa de altura entre o FlappyBird e a parte inferior do Pipe
    - Diferenзa de altura entre a parte inferior e superior do Pipe
    - Score

- Camadas de entrada diferentes para cada agente do AG:
    - Salva uma lista com todos os parametros para cada FlappyBird.
    - Na criaзгo da primeira populaзгo sorteia para cada FlappyBird uma quantidade de 1 - 6 de parвmetros (neurфnios de entrada) e depois sorteia os нndices de 0 - 5.


- Sorteia um nъmero de 1 - 100 para a quantidade de neurфnios na camada oculta.

- Camada de saнda fixa com 1 neurфnio.

- Funзгo de Fitness: Score (?)

- Seleзгo de indivнduos: Roleta ou Torneio (tamanho do torneio) (?)

- Reproduзгo: (?)

- Parвmetros de cada agente: 
    - n de neurфnios de entrada
    - indices dos neurфnios de entrada
    - n de neurфnios da camada oculta
    - learning rate
    - normalizador de input
    - bias
    - pesos de entrada
    - pesos de saнda