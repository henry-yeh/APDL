import torch
from utils import *
from train import train
import random
import numpy as np
from copy import copy, deepcopy

class Env:
    def __init__(self, opts=None) -> None:
        # nhead 2, d_hid 200, dropout 0.2, activation, optimizer, lr
        self.dim = 6
        self.dtype = ['int', 'int', 'float', 'int', 'int', 'float']
        self.range = [
            [1, 2, 4, 8, 16],
            [64, 128, 256, 512, 1024],
            [0.05, 0.1, 0.2, 0.4],
            [0, 1],
            [i for i in range(1, 12)],
            [5. ,1., 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
        ]
    
        self.activation_table = {
            0: 'relu',
            1: 'gelu'
        }

        self.optimizer_table = { 
            1: torch.optim.Adadelta,
            2: torch.optim.Adagrad,
            3: torch.optim.Adam,
            4: torch.optim.AdamW,
            5: torch.optim.Adamax,
            6: torch.optim.ASGD,
            7: torch.optim.NAdam,
            8: torch.optim.RAdam,
            9: torch.optim.RMSprop,
            10: torch.optim.Rprop,
            11: torch.optim.SGD
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataset = torch.load('./data/train_data.pt').to(self.device)
        self.val_dataset = torch.load('./data/val_data.pt').to(self.device)
        self.opts = opts

    def eval(self, ind):
        nhead = ind[0]
        d_hid = ind[1]
        dropout = ind[2]
        activation = self.activation_table[ind[3]]
        model = make_model(nhead, d_hid, dropout, activation).to(self.device)
        optimizer = self.optimizer_table[ind[4]](model.parameters(), lr=ind[5])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        best_model, best_val_ppl = train(model, self.opts, self.train_dataset, self.val_dataset, optimizer, scheduler)
        return best_val_ppl

    # def test_eval(self, ind):
    #     return sum(ind)
        



class GA:
    def __init__(self, env, pop_size, max_iter) -> None:
        self.env = env
        self.population = []
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.ppl = []
        self.mutation_rate = 0.1
        self.best_ind = None
        self.best_ppl = None

    def ini_population(self):
        for i in range(self.pop_size):
            ind = []
            for d in range(self.env.dim):
                ind.append(random.choice(self.env.range[d]))
            ppl =  self.env.eval(ind)
            self.population.append(ind)
            self.ppl.append(ppl)
        self.best_ppl = min(self.ppl)
        self.best_ind = self.population[np.argmin(np.array(self.ppl))]
    
    def crossover(self):
        new_pop = []
        match = np.arange(self.pop_size)
        np.random.shuffle(match)
        match = match.reshape(-1, 2)
        for match_id in range(self.pop_size//2):
            ind1 = self.population[match[match_id, 0]]
            ind2 = self.population[match[match_id, 1]]
            ind1, ind2 = self._crossover(copy(ind1), copy(ind2))
            new_pop.extend([ind1, ind2])
        return new_pop

    def _crossover(self, ind1, ind2):
        p1 = random.randint(0, len(ind1)-1)  # random includes both ends
        p2 = random.randint(p1, len(ind1)-1) # exchange p1-p2 (p1: p2+1) including both ends
        segment1 = ind1[p1: p2+1]
        segment2 = ind2[p1: p2+1]
        ind1[p1: p2+1] = segment2
        ind2[p1: p2+1] = segment1
        return ind1, ind2

    def mutation(self, new_pop):
        for i in range(len(new_pop)):
            new_pop[i] = self._mutation(new_pop[i])
        return new_pop
    
    def _mutation(self, ind):
        for d in range(len(ind)):
            if random.uniform(0, 1) < self.mutation_rate:
                ind[d] = random.choice(self.env.range[d])
        return ind
    
    def evaluation(self, new_pop):
        new_ppl = []
        for i in range(len(new_pop)):
            ppl = self.env.eval(new_pop[i])
            new_ppl.append(ppl)
        return new_ppl

    def selection(self, candidates, candidates_ppl):
        '''
        select among 2NP individuals 
        '''
        match = np.arange(2 * self.pop_size)
        np.random.shuffle(match)
        match = match.reshape(-1, 2)
        for comp in range(self.pop_size):
            if candidates_ppl[match[comp, 0]] < candidates_ppl[match[comp, 1]]:
                self.population[comp] = candidates[match[comp, 0]]
                self.ppl[comp] = candidates_ppl[match[comp, 0]]
            else:
                self.population[comp] = candidates[match[comp, 1]]
                self.ppl[comp] = candidates_ppl[match[comp, 1]]
    
    def update_best(self):
        self.best_ppl = min(self.ppl)
        self.best_ind = self.population[np.argmin(np.array(self.ppl))]
    
    def main_loop(self):
        self.ini_population()
        for i in range(self.max_iter):
            print(f'------ begin iteration {i} -------')
            new_pop = self.crossover()
            new_pop = self.mutation(new_pop)
            new_ppl = self.evaluation(new_pop)

            candidates = []
            candidates.extend(deepcopy(self.population))
            candidates.extend(new_pop)

            candidates_ppl = []
            candidates_ppl.extend(copy(self.ppl))
            candidates_ppl.extend(new_ppl)

            self.selection(candidates, candidates_ppl)

            self.update_best()

            print('record best:',)
            print(self.best_ind, ' with ppl : ', self.best_ppl)

if __name__ == '__main__':
    from options import get_options
    opts = get_options()
    env = Env(opts)
    ga = GA(env, pop_size=20, max_iter=50)
    ga.main_loop()