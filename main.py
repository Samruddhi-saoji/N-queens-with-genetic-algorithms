#faster but does not give all possible solutions

from copy import deepcopy
from random import random, shuffle
from numpy.random import randint
import time #to measure run time

########################################################

class Chromosome:
    def __init__(self, n) -> None:
        self.n = n
        self.genes = [] #list of genes
        # gene = integer btw [0,n)


    #fitness function
    #high fitness = better chromo
    def fitness(self):
        cols = set() #set of columns in which queen has been placed
        posi_diags = set() #set of diagonals in which queens ahse been placed
        neg_diags = set()

        c = 0 #collisions
        for i in range(0, self.n):
            #check collision in the columns
            if self.genes[i] in cols:
                c += 1
            else:
                cols.add(self.genes[i])

            #check collisions in the diagonal
            '''if (i + self.genes[i] in diags) and (abs(i - self.genes[i]) in diags):
                c += 2'''
            #if (i + self.genes[i] in diags) or (i - self.genes[i]) in diags:
            if i + self.genes[i] in posi_diags :
                c += 1
            else:
                posi_diags.add(i+self.genes[i])
            if i - self.genes[i] in neg_diags :
                c += 1
            else:
                neg_diags.add(i- self.genes[i])

        return (1/(1+c))
    #max value of fitness function = 1


    #print(chromosome) --> print the genes list
    def __repr__(self) -> str:
        return ''.join(str(gene) for gene in self.genes)



###########################################################


class GeneticAlgorithm:
    def __init__(self, n,  pop_size, tournament_size, crossover_rate ,mutation_rate) -> None:
        self.n = n
        self.pop_size = pop_size #same for all generations
        self.population = [] #the population
            #list of individuals in the pop

        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        ###### randomly initialise the 0th generation of population #######
        #there can be only 1 queen in each row and each column
        #thus all n genes in the chromosome must have a unique value
            #ie, each gene must be an int btw [0,n) and there can be no repition
        
        #sample chromosome [0,1,2,3,4..n-1] 
        sample = [i for i in range(0,n)]

        #each individual of the pop will be shuffled version of sample
        for _ in range(0, pop_size) :
            individual = Chromosome(self.n)

            shuffle(sample)
            individual.genes = deepcopy(sample)  #deepcopy is imp,
            #otherwise all the individual.genes will be reference to the same list "sample"

            self.population.append(individual)

        self.explored_states = set() #set of explored chromosomes
        self.solution = set() #set of valid solutions
        
        #calculate n factorial
        fact = 1
        for i in range(1,n+1):
            fact = fact*i
        self.n_factorial = fact



    #return the fittest individual in the population
    # pop = list of individuals in the population
    def get_fittest(self, pop) :
        fittest = pop[0]
        max = fittest.fitness() #max fitness found yet

        for individual in pop :
            fitness = individual.fitness()
            if fitness > max:
                #this is the fittest chromosome yet
                fittest = individual
                max = fitness

        return fittest



    #the actual algorithm
    def run(self):
        pop = self.population
        gen = 0 #generation number

        #while all possible states are not explored
        while len(pop) > 0 :
            #add each individual to explored states set
            for indi in pop:
                self.explored_states.add(tuple(indi.genes))

                if indi.fitness() == 1:
                    self.solution.add(tuple(indi.genes))
                    '''print(gen)
                    print(len(self.explored_states))'''
                    print()
            ##### selection and crossover #######
            next_gen = [] #list of individuals in next gen

            #select parents through tournament method and perform crossover
            x = max(int(len(pop)/10), 1)
            for i in range(0, self.pop_size):
                p1 = self.tournament(x, pop)
                child = self.crossover(p1)

                #mutate the child
                self.mutate(child)

                #if new, add child to next gen's population
                if tuple(child.genes) not in self.explored_states:
                    next_gen.append(child) 

            #next gen becomes the population
            pop = next_gen
            gen += 1
        #answers found
        
        self.print_answers()
        print("total number of solutions found is ", len(self.solution))
        print(f"Total generations is {gen}")


    #selection (tournament method)
        #tournament size = number of participants
    def tournament(self, tournament_size, pop):
        participants = [] 
        pop_size = len(pop)

        #select the particpants randomly from the population
        for _ in range(0, tournament_size):
            i = randint(pop_size)
            participants.append( pop[i] )

        #return the fittest participant
        return self.get_fittest(participants)



    #crossover  (1 parent 1 child approach)
    def crossover(self, p1):
        child = Chromosome(p1.n)
        prob = self.crossover_rate
        if random() < prob:

            #for two random indexes i1 and i2
            #reverse order of all elements btw i1 and i2

            i1 = randint(0,self.n)
            i2 = randint(0,self.n)

            if i1>i2:
                #exchange
                i1,i2 = i2, i1

            child.genes.extend(p1.genes[:i1])
            rem = deepcopy(p1.genes[i1:i2])
            rem.reverse()
            child.genes.extend(rem)
            child.genes.extend(p1.genes[i2:])

            return child
        
        return p1
    



    #mutation
    #swap the genes at 2 random indexes
    def mutate(self, chromosome):
        #change the gene mutation_rate% of times
        if random() < self.mutation_rate :
            i1 = randint(0,self.n)
            i2 = randint(0,self.n)
            chromosome.genes[i1] , chromosome.genes[i2] = chromosome.genes[i2] , chromosome.genes[i1]


    #display the board 
    def print_answers(self):
        for answer in self.solution:
            board = [[" " for x in range(self.n)] for y in range(self.n)]

            for i in range(0, len(answer)):
                board[i][answer[i]] = "Q"

            print(board)
            print()




###########################################################
#driver code
n = 8 #chromosome length

#hyper parameters
pop_size = 50
tournament_size = 5
crossover_rate = 0.75
mutation_rate = 0.25


start = time.time()
algo = GeneticAlgorithm(n, pop_size , tournament_size, crossover_rate , mutation_rate )
algo.run()
end = time.time()
tot = end - start
print("time=", tot)