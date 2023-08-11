import numpy as np
from random import random, choice
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridEnvironment:
    """
    Representa o ambiente em forma de grade (grid) onde o agente se move.
    """
    
    def __init__(self, w, h, stumble=0.1):
        """
        Inicializa o ambiente.
        
        :param w: Largura da grade.
        :param h: Altura da grade.
        :param stumble: Probabilidade do agente tropeçar e escolher uma ação aleatória.
        """
        self.width = w
        self.height = h
        self.stumble = stumble
        self.position = (0, 0)  # Posição inicial do agente.
        self.goal = (w-1, h-1)  # Posição objetivo.
        self.obstacles = [(2, 2), (3, 3), (4, 4)]  # Posições dos obstáculos.

    def reset(self):
        """
        Reinicia a posição do agente para a posição inicial.
        
        :return: Posição inicial do agente.
        """
        self.position = (0, 0)
        return self.position

    def act(self, action):
        """
        Move o agente com base na ação fornecida.
        
        :param action: Ação escolhida pelo agente ('up', 'down', 'left', 'right').
        :return: Nova posição do agente, recompensa e flag indicando se o episódio terminou.
        """
        if random() < self.stumble:
            action = choice(['up', 'down', 'left', 'right'])

        x, y = self.position
        if action == 'up' and y < self.height - 1:
            y += 1
        elif action == 'down' and y > 0:
            y -= 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < self.width - 1:
            x += 1

        self.position = (x, y)

        if self.position in self.obstacles:
            return self.position, -10, True
        elif self.position == self.goal:
            return self.position, 10, True
        else:
            return self.position, -1, False

class QLearning:
    """
    Implementa o algoritmo Q-Learning.
    """
    
    def __init__(self, w, h, alpha=0.2, epsilon=0.5, init=100):
        """
        Inicializa o algoritmo Q-Learning.
        
        :param w: Largura da grade.
        :param h: Altura da grade.
        :param alpha: Taxa de aprendizado.
        :param epsilon: Probabilidade de escolher uma ação aleatória (exploração).
        :param init: Valor inicial para a tabela Q.
        """
        self.q_table = np.full((w, h, 4), init, dtype=float)
        self.alpha = alpha
        self.epsilon = epsilon
        self.actions = ['up', 'down', 'left', 'right']

    def getAction(self, pos):
        """
        Escolhe uma ação para a posição atual usando a política epsilon-greedy.
        
        :param pos: Posição atual do agente.
        :return: Ação escolhida.
        """
        if random() < self.epsilon:
            return choice(self.actions)
        else:
            x, y = pos
            return self.actions[np.argmax(self.q_table[x, y])]

    def update(self, state, action, newstate, reward, final):
        """
        Atualiza a tabela Q com base na recompensa e na ação tomada.
        
        :param state: Estado atual.
        :param action: Ação tomada.
        :param newstate: Novo estado após a ação.
        :param reward: Recompensa recebida.
        :param final: Flag indicando se o episódio terminou.
        """
        x, y = state
        nx, ny = newstate
        action_idx = self.actions.index(action)
        max_future_q = np.max(self.q_table[nx, ny])
        current_q = self.q_table[x, y, action_idx]
        if final:
            new_q = reward
        else:
            new_q = (1 - self.alpha) * current_q + self.alpha * (reward + max_future_q)
        self.q_table[x, y, action_idx] = new_q

    def printQTable(self):
        """
        Imprime a tabela Q.
        """
        print(self.q_table)

    def printPolicy(self):
        """
        Imprime a política derivada da tabela Q.
        """
        policy = np.chararray((self.q_table.shape[0], self.q_table.shape[1]), itemsize=5, unicode=True)
        for i in range(self.q_table.shape[0]):
            for j in range(self.q_table.shape[1]):
                policy[i, j] = self.actions[np.argmax(self.q_table[i, j])]
        print(policy)
        return policy

def visualize_gridworld(map_instance, policy):
    """
    Visualiza o gridworld com obstáculos, ponto de partida, objetivo e política.
    
    :param map_instance: Instância do ambiente GridEnvironment.
    :param policy: Política derivada da tabela Q.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for x in range(map_instance.width + 1):
        ax.axvline(x, color='black', lw=2)
    for y in range(map_instance.height + 1):
        ax.axhline(y, color='black', lw=2)
    
    for obstacle in map_instance.obstacles:
        ax.add_patch(patches.Rectangle((obstacle[0], obstacle[1]), 1, 1, color='red'))
    
    ax.add_patch(patches.Circle((0.5, 0.5), 0.3, color='blue'))
    ax.add_patch(patches.Circle((map_instance.width - 0.5, map_instance.height - 0.5), 0.3, color='green'))
    
    for i in range(map_instance.width):
        for j in range(map_instance.height):
            if (i, j) not in map_instance.obstacles:
                if policy[j][i] == 'up':
                    plt.arrow(i + 0.5, j + 0.5, 0, 0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif policy[j][i] == 'down':
                    plt.arrow(i + 0.5, j + 0.5, 0, -0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif policy[j][i] == 'left':
                    plt.arrow(i + 0.5, j + 0.5, -0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif policy[j][i] == 'right':
                    plt.arrow(i + 0.5, j + 0.5, 0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

# Parâmetros do ambiente e do algoritmo.
R = 10
C = 10
episodes = 100000
mapa = GridEnvironment(w=C, h=R, stumble=0.1)
ql = QLearning(w=C, h=R, alpha=0.2, epsilon=0.5, init=100)

# Loop de treinamento.
for i in range(episodes):
    state = mapa.reset()
    for step in range(R * C):
        action = ql.getAction(state)
        newstate, reward, final = mapa.act(action)
        ql.update(state, action, newstate, reward, final)
        state = newstate
        if final:
            break

# Imprime os resultados.
ql.printQTable()
policy = ql.printPolicy()
visualize_gridworld(mapa, policy)