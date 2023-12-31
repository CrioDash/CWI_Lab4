import numpy as np


class WumpusWorld:

    # Кількість кроків виконана агентом
    steps = 0

    # Ініціалізація середовища
    # rows, columns: Розміри сітки
    # start_point: Початкова позиція агента (кортеж індексів рядка і стовпця)
    # pits: Список місць розташування ям
    # wumpus: Кортеж, що представляє місце розташування Вампуса
    # gold: Кортеж, що представляє місце розташування золота
    # episodes: Кількість епізодів для навчання агента
    # epsilon: Параметр компромісу між дослідженням і експлуатацією
    # discount_factor: Фактор зниження для майбутніх винагород
    # learning_rate: Коефіцієнт навчання для оновлення Q-таблиці
    def __init__(
            self,
            rows: int,
            columns: int,
            start_point: tuple,
            pits: list,
            wumpus: tuple,
            gold: tuple,
            episodes: int,
            epsilon: float,
            discount_factor: float,
            learning_rate: float) -> None:

        self.rows = rows
        self.rows = columns
        self.start_point = start_point
        self.pits = pits
        self.wumpus = wumpus
        self.gold = gold
        self.episodes = episodes
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.lr = learning_rate

    # Ініціалізує матрицю винагород і Q-таблицю на основі вказаних параметрів середовища
    def initialize(self) -> None:
        self.rewards = np.zeros((self.rows, self.rows))
        for pit in self.pits:
            self.rewards[pit] = -1
        self.rewards[self.wumpus] = -1
        self.rewards[self.gold] = 1
        self.q_table = np.zeros((self.rows, self.rows, 4))
        self.actions = ['up', 'down', 'right', 'left']

    # Перевіряє, чи заданий елемент сітки є термінальним станом (містить яму, Вампуса або золото)
    def is_terminal_state(self, row_index: int, col_index: int) -> bool:
        if self.rewards[row_index, col_index] == 0:
            return False
        else:
            return True

    # Вибирає наступну дію для агента за стратегією epsilon-greedy
    def next_action(self, current_row_index: int, current_col_index: int, epsilon: float) -> int:
        rand = np.random.random()
        if rand < epsilon:
            return np.argmax(self.q_table[current_row_index, current_col_index])
        else:
            return np.random.randint(len(self.actions))

    # Виконує вказану дію і повертає нове положення та пов'язану винагороду
    def move(self, current_row_index: int, current_col_index: int, action_index: int) -> tuple:
        new_row_index = current_row_index
        new_col_index = current_col_index
        if self.actions[action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1
        elif self.actions[action_index] == 'down' and current_row_index < self.rows - 1:
            new_row_index += 1
        elif self.actions[action_index] == 'right' and current_col_index < self.rows - 1:
            new_col_index += 1
        elif self.actions[action_index] == 'left' and current_col_index > 0:
            new_col_index -= 1

        reward = self.rewards[new_row_index, new_col_index]
        return new_row_index, new_col_index, reward

    # Оновлює Q-таблицю на основі досвіду агента
    def update_q_table(self, current_row: int, current_col: int,
                       action_index: int, reward: int, new_row: int, new_col: int) -> None:

        self.q_table[current_row,
        current_col,
        action_index] = self.q_table[current_row,
        current_col,
        action_index] * (1 - self.lr) + self.lr * \
                        (reward + self.discount_factor * np.max(self.q_table[new_row, new_col]))

    # Навчає агента, виконуючи епізоди, роблячи кроки та оновлюючи Q-таблицю
    def train(self) -> None:
        for episode in range(self.episodes):
            current_row, current_col = self.start_point
            while not self.is_terminal_state(current_row, current_col):
                self.steps += 1
                action_index = self.next_action(current_row, current_col, self.epsilon)
                new_row, new_col, reward = self.move(current_row, current_col, action_index)
                self.update_q_table(current_row, current_col, action_index, reward, new_row, new_col)
                current_row, current_col = new_row, new_col

    # Знаходить найкращий шлях від заданої точки використовуючи навчену Q-таблицю
    def best_path(self, start_point: tuple) -> list:
        start_row, start_col = start_point
        path = []
        if self.is_terminal_state(start_row, start_col):
            return path
        else:
            current_row, current_col = start_row, start_col
            path.append([current_row, current_col])
            while not self.is_terminal_state(current_row, current_col):
                action_index = self.next_action(current_row, current_col, 1)
                current_row, current_col, _ = self.move(current_row, current_col, action_index)
                path.append([current_row, current_col])
        return path
