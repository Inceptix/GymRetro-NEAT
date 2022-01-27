import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

imgarray = []

xpos_end = 0


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx / 8)
        iny = int(iny / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        cv2.namedWindow("train_view", cv2.WINDOW_NORMAL)            # what the network sees

        while not done:

            env.render()  # turn off this line to run in the background
            frame += 1

            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGRA2GRAY)            # what the network sees
            scaledimg = cv2.resize(scaledimg, (iny, inx))                # what the network sees

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            cv2.imshow('train_view', scaledimg)                     # what the network sees
            cv2.waitKey(1)                                          # what the network sees

            imgarray = np.ndarray.flatten(ob)
            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)

            xpos = info['x']
            xpos_end = info['screen_x_end'] + 320
            screen_x_right = info['screen_x'] + 320

            if xpos > xpos_max:
                fitness_current += rew
                xpos_max = xpos
                counter -= 1
                #print(fitness_current, xpos, xpos_max, screen_x_right)
            else:
                counter += 2
                #print("counter += 1")

            if xpos >= xpos_end and xpos > 500:
                fitness_current += 100000
                done = True

            #fitness_current += rew

            if frame == 7200:
                done = True
                print(genome_id, fitness_current)

            #if fitness_current > current_max_fitness:
                #current_max_fitness = fitness_current
                #counter = 0
                #print("counter = 0", "current ", fitness_current, "max ", current_max_fitness)
            #elif (current_max_fitness * 0.95) > xpos:
                #counter += 2
                #print("count += 5")
            #else:
                #counter += 1
                #print("counter += 1")

            if done or counter >= (xpos*0.6)+100:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(20))

winner = p.run(eval_genomes)

with open('winner_mine.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
