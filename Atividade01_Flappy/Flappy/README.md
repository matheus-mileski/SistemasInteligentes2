# Criar venv e instalar requirements
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```
# Testar o MLP com o Iris

```
python TesteMLP.py
```

# Executar Flappy com AG e MLP

```
python AT01-SI2-MainFlappy.py
```

# Ideias Iniciais

### Possíveis Entradas do MLP:
- Posição Vertical do FlappyBird
- Velocidade Vertical do FlappyBird
- Distância pro próximo Pipe
- Diferença de altura entre o FlappyBird e a parte inferior do Pipe
- Diferença de altura entre a parte inferior e superior do Pipe
- Score

### Camadas de Entrada Diferentes para Cada Agente do AG:
- Salva uma lista com todos os parâmetros para cada FlappyBird.
- Na criação da primeira população, sorteia para cada FlappyBird uma quantidade de 1 - 6 de parâmetros (neurônios de entrada) e depois sorteia os índices de 0 - 5.

### Sorteia um Número de Neurônios na Camada Oculta:
- Sorteia um número de 1 - 1000 para a quantidade de neurônios na camada oculta.

### Camada de Saída:
- Camada de saída fixa com 1 neurônio.

### Função de Fitness:
- Função de Fitness: Score + (Máximo de Gerações do MLP - Geração final)
    - Dessa forma bonificamos os agentes que chegam mais rápido na pontuação máxima.

### Seleção de Indivíduos:
- Seleção de indivíduos: Torneio (tamanho do torneio = 3)

### Reprodução:
- Reprodução: Crossover com 3 pais

### Parâmetros de Cada Agente: 
- n de neurônios de entrada
- índices dos neurônios de entrada
- n de neurônios da camada oculta
- learning rate
- normalizador de input
- bias
- pesos de entrada
- pesos de saída

# Algoritmo Genético para Treinar uma Rede Neural no Jogo Flappy Bird

## Estrutura Genômica
- **GENOME:** Lista com parâmetros que representam a estrutura e comportamento da rede neural MLP.
    - **n_input:** Número de neurônios de entrada.
    - **i0, i1, i2, i3, i4, i5:** Flags indicando quais entradas estão ativas.
    - **n_hidden:** Número de neurônios na camada oculta.
    - **learn_rate:** Taxa de aprendizado para o treinamento da rede neural.
    - **norm:** Fator de normalização.

## Seleção de Indivíduos: Método do Torneio
- O método de seleção por torneio escolhe um subconjunto de indivíduos aleatoriamente da população e seleciona o melhor entre eles para se tornar um pai.
- Este processo é repetido para selecionar todos os pais necessários.

### Características do Método do Torneio
- **Seleção Estocástica:** Mesmo os indivíduos com baixo desempenho têm uma chance de serem selecionados.
- **Pressão Seletiva:** O tamanho do torneio controla a pressão seletiva. Torneios maiores aumentam a chance de indivíduos de alta adequação serem selecionados.

## Reprodução: Crossover e Mutação
### Crossover: Tipo Uniforme
- **Crossover Uniforme:** Pares de pais são combinados gerando descendentes, onde os genes dos descendentes são uma mistura uniforme dos genes dos pais.
- Para cada gene, um pai é escolhido aleatoriamente para doar seu gene para o descendente.
- Este método permite que todos os genes tenham uma chance igual de serem transmitidos para a próxima geração.

### Mutação: Tipo Swap
- **Mutação Swap:** Troca a posição de dois genes aleatórios dentro de um genoma.
- Ajuda a introduzir variação na população trocando aleatoriamente os genes.
- É crucial para manter a diversidade na população e evitar a convergência prematura para um mínimo local.

## Estratégias de Evolução
- **Inicialização:** A população inicial é gerada aleatoriamente, garantindo uma ampla exploração do espaço de busca.
- **Seleção:** Indivíduos são selecionados com base em sua aptidão para serem pais usando o método de seleção por torneio.
- **Crossover:** Pares de pais são cruzados para criar descendentes usando o crossover uniforme.
- **Mutação:** Variação é introduzida na população descendente através da mutação swap.
- **Avaliação:** A adequação de cada indivíduo na população é avaliada usando a função de adequação.
- **Seleção de Sobreviventes:** Indivíduos são selecionados para formar a próxima geração.
- Este ciclo de seleção, crossover, mutação e avaliação continua até que um critério de parada seja atingido (por exemplo, número máximo de gerações).

## Métodos e suas Funções
1. **`__init__`:** Inicializa os parâmetros básicos do algoritmo genético.
   - **num_generations:** Número de gerações.
   - **population_size:** Tamanho da população.
   - **num_parents:** Número de pais selecionados para crossover.
   - **initial_population:** Chama o método `generate_initial_population` para criar a população inicial.
   - **ga_instance:** Instância do algoritmo genético usando `pygad.GA`.

2. **`generate_initial_population`:** Gera a população inicial.
   - **pop_size:** Tamanho da população a ser gerada.
   - Retorna uma matriz numpy da população inicial.

3. **`generate_genome`:** Gera um genoma com valores aleatórios dentro dos limites especificados.
   - Retorna uma lista contendo o genoma gerado.

4. **`fitness_function`:** Calcula a adequação de uma solução.
   - **ga_instance:** Instância atual do algoritmo genético.
   - **solution:** A solução a ser avaliada.
   - **solution_idx:** Índice da solução.
   - Retorna a pontuação (adequação) da solução.

5. **`decode_genome`:** Decodifica um genoma em parâmetros utilizáveis para a rede neural.
   - **genome:** O genoma a ser decodificado.
   - Retorna um dicionário com os parâmetros decodificados.

6. **`encode_genome`:** Codifica parâmetros da rede neural em um genoma.
   - **params:** Dicionário de parâmetros a serem codificados.
   - Retorna uma lista contendo o genoma codificado.

7. **`ensure_valid_genomes`:** Garante que os genomas na população aderem às restrições do problema.
   - **ga_instance:** Instância atual do algoritmo genético.
   - **offspring:** A prole gerada após a operação de crossover.
   - Ajusta genomas na população.

8. **`run_flappy_bird`:** Executa uma instância do Flappy Bird usando a rede neural e parâmetros fornecidos.
   - **mlp_params:** Parâmetros para a rede neural.
   - **generation:** Geração atual.
   - **max_generations:** Máximo de gerações.
   - **solution_idx:** Índice da solução.
   - Retorna a pontuação alcançada pela rede neural no jogo.

9. **`run`:** Executa o algoritmo genético.
   - Retorna a melhor solução e sua adequação.

## Execução
- Quando executado como o script principal, uma instância da classe `FlappyBirdGA` é criada e executada.
