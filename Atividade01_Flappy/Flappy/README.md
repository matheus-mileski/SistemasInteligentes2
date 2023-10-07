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

- Poss�veis entradas do MLP:
    - Posi��o Vertical do FlappyBird
    - Velocidade Vertical do FlappyBird
    - Dist�ncia pro pr�ximo Pipe
    - Diferen�a de altura entre o FlappyBird e a parte inferior do Pipe
    - Diferen�a de altura entre a parte inferior e superior do Pipe
    - Score

- Camadas de entrada diferentes para cada agente do AG:
    - Salva uma lista com todos os parametros para cada FlappyBird.
    - Na cria��o da primeira popula��o sorteia para cada FlappyBird uma quantidade de 1 - 6 de par�metros (neur�nios de entrada) e depois sorteia os �ndices de 0 - 5.


- Sorteia um n�mero de 1 - 100 para a quantidade de neur�nios na camada oculta.

- Camada de sa�da fixa com 1 neur�nio.

- Fun��o de Fitness: Score (?)

- Sele��o de indiv�duos: Roleta ou Torneio (tamanho do torneio) (?)

- Reprodu��o: (?)

- Par�metros de cada agente: 
    - n� de neur�nios de entrada
    - indices dos neur�nios de entrada
    - n� de neur�nios da camada oculta
    - bias
    - pesos de entrada
    - pesos de sa�da