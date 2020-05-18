from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent

for i in range(0,1):
    game = Game(4, random=False,score_to_win=2048)

    agent = ExpectiMaxAgent(game )
    agent.play(verbose=False,save=True,savedir='data/2048.txt')
    print(str(i+1)+"th game completed")