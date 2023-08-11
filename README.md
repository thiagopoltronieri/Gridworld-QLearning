# Projeto de Aprendizagem por Reforço com Q-Learning

Este projeto implementa um algoritmo de aprendizagem por reforço chamado Q-Learning em um ambiente de grid. O agente se move pelo grid, evitando obstáculos e tentando alcançar o objetivo. Após a execução, o programa utiliza a biblioteca `matplotlib` para exibir visualmente a tabela Q e a política derivada, mostrando as ações que o agente deve tomar em cada posição para maximizar sua recompensa.

## Referências

Este projeto foi inspirado e baseado nas seguintes fontes:

1. **Livro**: "Reinforcement Learning: An Introduction" por Richard S. Sutton e Andrew G. Barto. 
   - [Link para o livro](http://incompleteideas.net/book/RLbook2020.pdf)

2. **Vídeo**: "[AULA] Aprendizagem por Reforço e o Algoritmo Q-Learning" do Canal SorPinto.
   - [Link para o vídeo](https://youtu.be/tz8phEIqKAM)

## Parâmetros

- **R e C**: Definem as dimensões do ambiente de grade.
- **episodes**: Número de episódios para treinamento.
- **stumble**: Probabilidade do agente tomar uma ação aleatória (exploração).
- **alpha**: Taxa de aprendizado.
- **epsilon**: Fator de exploração.
- **init**: Valor inicial para a tabela Q.

## Estrutura do Código

O código é composto por duas classes principais:

1. `GridEnvironment`: Representa o ambiente em forma de grid onde o agente se move.
2. `QLearning`: Implementa o algoritmo Q-Learning para treinar o agente. Após a execução, utiliza a biblioteca `matplotlib` para visualizar a tabela Q e a política derivada.

## Como Rodar

1. **Clone o Repositório**:
   ```bash
   git clone https://github.com/thiagopoltronieri/Gridworld-QLearning.git
   cd Gridworld-QLearning
   ```

2. **Instale as Dependências**:
   ```bash
   pip install numpy matplotlib
   ```

3. **Execute o Código**:
   Abra o projeto no VSCode ou em seu editor de código favorito. Execute o arquivo main.py para ver o algoritmo em ação e visualizar a tabela Q e a política derivada utilizando `matplotlib`.