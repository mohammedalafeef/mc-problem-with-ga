import sys
import random
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import deque, Sequence, Counter
from numbers import Integral
from enum import IntEnum, Enum, auto


class Group():
    """Group of missionaries and cannibals"""
    def __init__(self, missionaries, cannibals):
        self.reset(missionaries, cannibals)
    
    def reset(self, missionaries, cannibals):
        self.missionaries = missionaries
        self.cannibals = cannibals
    
    def __str__(self):
        return  f'(m:{self.missionaries}, c:{self.cannibals})'
    
    def __add__(self, other):
        assert isinstance(other, Group)
        return Group(self.missionaries + other.missionaries,
                     self.cannibals + other.cannibals)
    
    def __sub__(self, other):
        assert isinstance(other, Group)
        return Group(self.missionaries - other.missionaries,
                     self.cannibals - other.cannibals)
    
    def is_valid(self):
        return 0 <= self.missionaries <= 3 and 0 <= self.cannibals <= 3
    
    def is_capacity_over(self):
        return not (1 <= (self.missionaries + self.cannibals) <= 2)
    
    def is_safe(self):
        return self.missionaries == 0 or self.missionaries >= self.cannibals
    
    def is_empty(self):
        return self.missionaries == self.cannibals == 0


class MCProblem():
    """Missionaries and cannibals problem"""
    _LEFT_BANK = 0
    _RIGHT_BANK = 1
    
    class Result(Enum):
        ILLEGAL_MOVE = auto()
        CAPACITY_OVER = auto()
        EATEN_BY_CANNIBALS = auto()
        REPETITION = auto()
        LEGAL_MOVE = auto()
        GAME_CLEAR = auto()
    
    def __init__(self):
        self.searched_nodes = 0
        self.reset()
    
    def reset(self):
        self._left_bank = Group(3, 3)
        self._right_bank = Group(0, 0)
        self._boat_position = self._LEFT_BANK
        self._records = [self._record()]
    
    def _record(self):
        return hash((str(self._left_bank),
                     str(self._right_bank), self._boat_position))
    
    _boat = {_LEFT_BANK : '=  ', _RIGHT_BANK: '  ='}
    
    def __str__(self):
        boat = self._boat[self._boat_position]
        left_bank = str(self._left_bank)
        right_bank = str(self._right_bank)
        return left_bank + boat + right_bank
    
    def _forward(self, passengers, verbose):
        self._left_bank -= passengers
        self._right_bank += passengers
        self._boat_position = self._RIGHT_BANK
        if verbose:
            print(f'    ==(m:{passengers.missionaries},'
                  f' c:{passengers.cannibals})=>')
    
    def _backward(self, passengers, verbose):
        self._left_bank += passengers
        self._right_bank -= passengers
        self._boat_position = self._LEFT_BANK
        if verbose:
            print(f'    <=(m:{passengers.missionaries},'
                  f' c:{passengers.cannibals})==')
    
    _cross_the_river = {_LEFT_BANK : _forward,
                        _RIGHT_BANK: _backward}
    
    def _move(self, passengers, verbose):
        self._cross_the_river[self._boat_position](self, passengers, verbose)
        
        if not (self._left_bank.is_valid() and self._right_bank.is_valid()):
            return self.Result.ILLEGAL_MOVE
        
        if passengers.is_capacity_over():
            return self.Result.CAPACITY_OVER
        
        if not (self._left_bank.is_safe() and self._right_bank.is_safe()):
            return self.Result.EATEN_BY_CANNIBALS
        
        if self._record() in self._records:
            return self.Result.REPETITION
        
        if self._left_bank.is_empty():
            return self.Result.GAME_CLEAR
        
        self._records.append(self._record())
        return self.Result.LEGAL_MOVE
    
    _countup = {Result.ILLEGAL_MOVE: 0, Result.CAPACITY_OVER: 0,
                Result.EATEN_BY_CANNIBALS: 0, Result.REPETITION: 0,
                Result.LEGAL_MOVE: 1, Result.GAME_CLEAR: 1}
    
    def moves(self, missionaries, cannibals, verbose=False):
        assert len(missionaries) == len(cannibals)
        self.reset()
        legal_moves = 0
        for i in range(len(missionaries)):
            if verbose: print(self)
            result = self._move(Group(missionaries[i], cannibals[i]), verbose)
            self.searched_nodes += 1
            legal_moves += self._countup[result]
            if result == self.Result.LEGAL_MOVE:
                continue
            break
        if verbose: print(self)
        return result, legal_moves


class Individual():
    """Individual that stands for the answer of the problem"""
    _GENE_SIZE = 80
    
    _MASK_UINT2 = 0b11
    _SHIFT_UINT2 = 2
    
    _MUTATE_PROBABILITY = 0.05
    
    def __init__(self, mc_problem, gene=None):
        if gene is None:
            self.gene = random.getrandbits(self._GENE_SIZE)
        else:
            assert 0 <= gene < 2 ** self._GENE_SIZE
            self.gene = gene
        self._mc_problem = mc_problem
        
        # at first, we are trying to parse a gene
        gene = self.gene
        
        self.missionaries = [0] * (self._GENE_SIZE // 4)
        self.cannibals = [0] * (self._GENE_SIZE // 4)
        
        for i in range(self._GENE_SIZE // 4):
            self.missionaries[i] = (gene & self._MASK_UINT2)
            gene = gene >> self._SHIFT_UINT2
        
        for i in range(self._GENE_SIZE // 4):
            self.cannibals[i] = (gene & self._MASK_UINT2)
            gene = gene >> self._SHIFT_UINT2
        
        # next, calculate the fitness
        self.calc_fitness()
    
    def _formula1(self):
        return 1. - (1. / 1.4 ** self.legal_moves)
    
    def _formula2(self):
        return 1.
    
    _fitness_from = {MCProblem.Result.ILLEGAL_MOVE      : _formula1,
                     MCProblem.Result.CAPACITY_OVER     : _formula1,
                     MCProblem.Result.EATEN_BY_CANNIBALS: _formula1,
                     MCProblem.Result.REPETITION        : _formula1,
                     MCProblem.Result.LEGAL_MOVE        : _formula1,
                     MCProblem.Result.GAME_CLEAR        : _formula2}
    
    def calc_fitness(self):
        self.result, self.legal_moves = \
        self._mc_problem.moves(self.missionaries, self.cannibals)
        self.fitness = self._fitness_from[self.result](self)
    
    def _is_valid_operand(self, other):
        return hasattr(other, 'fitness')
    
    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness == other.fitness
    
    def __ne__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness != other.fitness
    
    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness < other.fitness
    
    def __le__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness <= other.fitness
    
    def __gt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness > other.fitness
    
    def __ge__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness >= other.fitness
    
    def __str__(self):
        return f'gene: {self.gene}'
    
    def mate(self, other, mc_problem):
        """Give birth to a child"""
        assert isinstance(other, Individual)
        
        child_gene = 0
        self_gene = self.gene
        other_gene = other.gene
        
        # inherits from parents
        mask_mate = random.getrandbits(self._GENE_SIZE)
        self_gene = self_gene & mask_mate
        other_gene = other_gene & ~mask_mate
        child_gene = self_gene | other_gene
        
        # genetic mutation
        mask_mutation = 0
        for _ in range(self._GENE_SIZE):
            mask_mutation = mask_mutation << 1
            if random.random() <= self._MUTATE_PROBABILITY:
                mask_mutation = mask_mutation | 0b1
        child_gene = child_gene ^ mask_mutation
        
        return Individual(mc_problem, child_gene)


class Population():
    """Group of individuals"""
    _FERTILITY_RATE = 10
    _CATASTOROPHE_DAMAGE = 100
    
    def __init__(self, mc_problem, population_size):
        self._mc_problem = mc_problem
        self._population_size = population_size
        self.generation = [Individual(self._mc_problem)
                              for _ in range(self._population_size)]
        self.generation.sort(reverse=True)
        self.generation_number = 0
    
    def next_generation(self):
        self.generation_number += 1
        
        # divide individuals into elites and non-elites
        pareto = self._population_size // 5
        elites = self.generation[: pareto]
        non_elites = self.generation[pareto :]
        
        # all the elite to have a chance to marry non-elite
        children = []
        for parent1 in elites:
            parent2 = random.choice(non_elites)
            for _ in range(self._FERTILITY_RATE):
                children.append(parent1.mate(parent2, self._mc_problem))
        
        # choose individuals to survive
        elites = random.sample(elites, 12 * len(elites) // 15)
        non_elites = random.sample(non_elites, 3 * len(non_elites) // 15)
        self.generation = elites + children + non_elites
        self.generation.sort(reverse=True)
        self.generation = self.generation[: self._population_size]
        
        # logging the generation
        min_fitness = self.generation[-1].fitness
        max_fitness = self.generation[0].fitness
        mean_fitness = mean(i.fitness for i in self.generation)
        median_fitness = self.generation[self._population_size // 2].fitness
        
        return Aspect(min_fitness, max_fitness, mean_fitness, median_fitness)
    
    def catastrophe(self):
        """Some kind of natural disaster that would cause a wider evolution"""
        survivor_num = self._population_size // self._CATASTOROPHE_DAMAGE
        survivor = random.sample(self.generation, survivor_num)
        newcomer = [Individual(self._mc_problem)
                        for _ in range(self._population_size)]
        self.generation = survivor + newcomer
        self.generation.sort(reverse=True)
        self.generation = self.generation[: self._population_size]


class Aspect():
    """Aspect of the evolution"""
    def __init__(self, min, max, mean, median):
        self.min = min
        self.max = max
        self.mean = mean
        self.median = median


class Data4Graph():
    """Store of the data for drawing a graph"""
    def __init__(self):
        self.min = []
        self.max = []
        self.mean = []
        self.median = []
    
    def append(self, aspect):
        assert isinstance(aspect, Aspect)
        self.min.append(aspect.min)
        self.max.append(aspect.max)
        self.mean.append(aspect.mean)
        self.median.append(aspect.median)
    
    def check(self):
        return (len(self.min)
                == len(self.max) == len(self.mean) == len(self.median))


def visualize(data4graph):
    """Draw a graph"""
    assert isinstance(data4graph, Data4Graph)
    assert data4graph.check()
    x = range(1, len(data4graph.min) + 1)
    plt.figure()
    plt.plot(x, data4graph.min, marker='.', label='min_fitness')
    plt.plot(x, data4graph.max, marker='.', label='max_fitness')
    plt.plot(x, data4graph.mean, marker='.', label='mean_fitness')
    plt.plot(x, data4graph.median, marker='.', label='median_fitness')
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=10)
    plt.grid()
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('Missionaries and cannibals problem')
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()


def is_ipython():
    try:
        get_ipython()
    except:
        result = False
    else:
        result = True
    return result


class OutputManager():
    """Managing output to the stdin"""
    def __init__(self, verbose):
        self._verbose = verbose
        self._ipython = is_ipython()
    
    _arrow = {0: '\r↑', 1: '\r→', 2: '\r↓', 3: '\r←'}
    
    def _clear_output(self):
        if self._ipython:
            # carriage return only
            s = '\r'
        else:
            # erase in line and carriage return
            s = '\033[2K\033[G'
        sys.stdout.write(s)
        sys.stdout.flush()
    
    def success(self, generation_number, aspect_max, legal_moves,
                searched_nodes, answer_gene, missionaries, cannibals):
        if self._verbose == 0:
            self._clear_output()
        print(f'{generation_number}: {aspect_max}({legal_moves})')
        print(f'searched_nodes: {searched_nodes}')
        print(answer_gene)
        print(f'm: {missionaries}')
        print(f'c: {cannibals}')
    
    def continue_(self, generation_number, aspect_max, legal_moves):
        if self._verbose == 0:
            if not self._ipython: self._clear_output()
            if self._ipython:
                s = self._arrow[generation_number % 4]
            else:
                s = f'{generation_number}: {aspect_max}({legal_moves})'
            sys.stdout.write(s)
            sys.stdout.flush()
        elif self._verbose > 0:
            print(f'{generation_number}: {aspect_max}({legal_moves})')
    
    def catastrophe(self):
        if self._verbose == 0:
            if self._ipython:
                s = '\r※'
            else:
                self._clear_output()
                s = 'CATASTROPHE OCCURED!'
            sys.stdout.write(s)
            sys.stdout.flush()
        elif self._verbose > 0:
            print('CATASTROPHE OCCURED!')
    
    def epoch_over(self, generation_number, aspect_max, legal_moves):
        if self._verbose == 0:
            self._clear_output()
            print(f'{generation_number}: {aspect_max}({legal_moves})')


class EvolutionController():
    """Some existence that controlls the evolution"""
    def __init__(self, mc_problem, population_size=100, epochs=1000,
                 patience=30, verbose=0, graph=False):
        self._mc_problem = mc_problem
        self._population = Population(self._mc_problem, population_size)
        self._epochs = epochs
        self._patience = patience 
        self._memory = deque([], patience)
        self._graph = graph
        if self._graph: self._data4graph = Data4Graph()
        self._outmgr = OutputManager(verbose)
    
    def start(self):
        """Start the evolution"""
        for epoch in range(1, self._epochs + 1):
            aspect = self._population.next_generation()
            top1 = self._population.generation[0]
            if aspect.max == 1.:
                answer = (top1.missionaries[:top1.legal_moves],
                          top1.cannibals[:top1.legal_moves])
                self._outmgr.success(self._population.generation_number,
                                     aspect.max, top1.legal_moves,
                                     self._mc_problem.searched_nodes,
                                     str(top1), *answer)
                # print the answer
                self._mc_problem.reset()
                self._mc_problem.moves(*answer, verbose=True)
                
                if self._graph:
                    self._data4graph.append(aspect)
                    visualize(self._data4graph)
                break
            else:
                self._outmgr.continue_(self._population.generation_number,
                                       aspect.max, top1.legal_moves)
                if self._graph:
                    self._data4graph.append(aspect)
                
                # catastrophe check
                self._memory.append(aspect.max)
                if self._memory.count(self._memory[-1]) == self._patience:
                    self._outmgr.catastrophe()
                    self._population.catastrophe()
        else:
            self._outmgr.epoch_over(self._population.generation_number,
                                    aspect.max,
                                    self._population.generation[0].legal_moves)
            if self._graph:
                visualize(self._data4graph)


def moves_for_0th_gen(n=10):
    mc_problem = MCProblem()
    xs = [Individual(mc_problem) for _ in range(n)]
    for x in xs:
        result, _ = mc_problem.moves(x.missionaries, x.cannibals, verbose=True)
        print(result)
        print(f'------------------')


def stat_for_0th_gen(n=100):
    mc_problem = MCProblem()
    xs = [Individual(mc_problem) for _ in range(n)]
    legal_moves = Counter([x.legal_moves for x in xs])
    results = Counter([x.result for x in xs])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('Legal moves')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.pie(legal_moves.values(), labels=legal_moves.keys(),
            autopct='%.1f', pctdistance=0.6, labeldistance=1.1,
            startangle=90, frame=False)
    ax1.axis('equal')
    if not is_ipython(): fig1.show()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('Result')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.pie(results.values(), labels=results.keys(),
            autopct='%.1f', pctdistance=0.6, labeldistance=1.1,
            startangle=90, frame=False)
    ax2.axis('equal')
    if not is_ipython(): fig2.show()


def main():
    mc_problem = MCProblem()
    ec = EvolutionController(mc_problem, verbose=0, graph=True)
    ec.start()


if __name__ == '__main__':
    main()
