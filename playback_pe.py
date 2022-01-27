import retro
import numpy as np
import cv2 
import neat
import pickle
import time


env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record='.')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')


# p = neat.Population(config)


# p.add_reporter(neat.StdOutReporter(True))
# stats = neat.StatisticsReporter()
# p.add_reporter(stats)

with open('winner_parallel_mine.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()

inx, iny, _ = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

done = False

while not done:
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    ob = ob.ravel()
    ob = np.interp(ob,(0, 254), (-1, +1))

    nnOutput = net.activate(ob)

    ob, rew, done, info = env.step(nnOutput)

    xpos = info['x']
    
    if xpos > xpos_max:
        xpos_max = xpos
        fitness_current += 1
    
    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter += 1
    
    if counter == 250:
        done = True
print(fitness_current)