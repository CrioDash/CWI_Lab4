from wumpus import WumpusWorld

ENVIRONMENT_ROWS = 4
ENVIRONMENT_COLUMNS = 4
START_POINT = (0, 0)
PITS = [(0, 3), (1, 2), (3, 2)]
WUMPUS = (1, 0)
GOLD = (3, 3)
EPISODES = 2000
EPSILON = 0.3
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.8

wumpus_agent = WumpusWorld(ENVIRONMENT_ROWS, ENVIRONMENT_COLUMNS, START_POINT, PITS, WUMPUS, GOLD,
                           EPISODES, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE)

wumpus_agent.initialize()
print('Ігрове поле')
print(wumpus_agent.rewards)
print('-------------------------------------------')
print('Кількість епізодів тренування: ', wumpus_agent.episodes)
print('Параметр компромісу між дослідженням і експлуатацією: ', wumpus_agent.epsilon)
print('Фактор зниження для майбутніх винагород: ', wumpus_agent.discount_factor)
print('Початок тренування')
print('-------------------------------------------')
wumpus_agent.train()
print('Тренування закінчене')
print('Кількість кроків: ', wumpus_agent.steps)
print("Шлях до золота")
print(*wumpus_agent.best_path(START_POINT))
