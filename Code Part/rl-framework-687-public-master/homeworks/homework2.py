import numpy as np
from rl687.environments.gridworld import Gridworld
from rl687.environments.cartpole import Cartpole
from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
from rl687.agents.ga import GA
from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.tabular_softmax_continuous import TabularSoftmaxContinuous
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42  # avoid type 3 fonts
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot(returns, title, red_line):
    
    episodes = np.array([i for i in range(returns.shape[1])])
    returns_mean = np.mean(returns, axis = 0)
    returns_std = np.std(returns, axis = 0)
    plt.errorbar(episodes, returns_mean, color='darkblue') #yerr=returns_std, 
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Expected Return')
    plt.plot([0, episodes.size - 1], [red_line, red_line], color="red")
    plt.fill_between(episodes, returns_mean + returns_std, returns_mean - returns_std, color='violet')
    plt.show()

def problem1():
    """
    Apply the CEM algorithm to the More-Watery 687-Gridworld. Use a tabular 
    softmax policy. Search the space of hyperparameters for hyperparameters 
    that work well. Report how you searched the hyperparameters, 
    what hyperparameters you found worked best, and present a learning curve
    plot using these hyperparameters, as described in class. This plot may be 
    over any number of episodes, but should show convergence to a nearly 
    optimal policy. The plot should average over at least 500 trials and 
    should include standard error or standard deviation error bars. Say which 
    error bar variant you used. 
    """

    #TODO
    
    popSize = 10 #10
    numElite = 5 #5
    epsilon = 4.0 #4.0
    sigma = 1.0 #1.0
    numEpisodes = 20 #50
    numTrials = 5 #5
    numIterations = 50 #200

    returns = np.zeros((numTrials, numEpisodes * numIterations))
    
    for trial in range(numTrials):
        

        np.random.seed(np.random.randint(10000))
    
        gridworld = Gridworld()

        tabular_softmax = TabularSoftmax(25, 4)
        theta = np.random.randn(tabular_softmax.parameters.shape[0])
        
        count = 0

        def evaluateFunction(theta, numEpisodes):
            nonlocal count
0
            expected_reward = 0

            numTimeSteps = 10000
            tabular_softmax.parameters = theta

            for episode in range(numEpisodes):
                state = gridworld.state
                G = 0
                discount = 1
                for t in range(numTimeSteps):
                    action = tabular_softmax.samplAction(state);
                    nextstate, reward, end = gridworld.step(action)
                    G += (discount) * reward
                    discount *= gridworld.gamma
                    if end == True:
                        break
                    elif t == 200:
                        G = -50
                        break
                    state = nextstate
                expected_reward += G
                returns[trial][count] = G
                gridworld.reset()
                count += 1

            return expected_reward / numEpisodes


        agent = CEM(theta, sigma, popSize, numElite, numEpisodes, evaluateFunction, epsilon)

        

        for iteration in range(numIterations):
        
            print("Trial: %d" % (trial, ))
            print("Iteration: %d" % (iteration, ))
            p = agent.train()
            l = [[0 for i in range(5)] for j in range(5)] 
            for i in range(25):
                k = tabular_softmax.getActionProbabilities(i)
                print(k)
                r = np.argmax(k)
                if(r == 0):
                    l[i//5][i % 5] = '↑'
                elif(r == 1):
                    l[i//5][i % 5] = '↓'
                elif(r == 2):
                    l[i//5][i % 5] = '←'
                elif(r == 3):
                    l[i//5][i % 5] = '→'

            for i in range(5):
                print(l[i])
        print(p)
            
    plot(returns, 'More-Watery 687-Gridworld domain Cross Entropy Method (standard deviation error bars) - 5 trials', 3)

def problem2():
    """
    Repeat the previous question, but using first-choice hill-climbing on the 
    More-Watery 687-Gridworld domain. Report the same quantities.
    """
    
    #TODO
    
    
    sigma = 1.0 #1.0

    numEpisodes = 200 #200
    numTrials = 50 #50
    numIterations = 200 #200

    returns = np.zeros((numTrials, numEpisodes * numIterations))
    
    for trial in range(numTrials):
        

        np.random.seed(np.random.randint(10000))
    
        gridworld = Gridworld()

        tabular_softmax = TabularSoftmax(25, 4)
        theta = np.random.randn(tabular_softmax.parameters.shape[0])
        
        count = -1


        def evaluateFunction(theta, numEpisodes):
            nonlocal count

            expected_reward = 0

            numTimeSteps = 10000
            tabular_softmax.parameters = theta

            for episode in range(numEpisodes):
                state = gridworld.state
                G = 0
                discount = 1
                for t in range(numTimeSteps):
                    action = tabular_softmax.samplAction(state);
                    nextstate, reward, end = gridworld.step(action)
                    G += (discount) * reward
                    discount *= gridworld.gamma
                    if end == True:
                        break
                    elif t == 100:
                        break
                    state = nextstate
                expected_reward += G
                if(count != -1):
                    returns[trial][count] = G
                    count += 1
                gridworld.reset()

            return expected_reward / numEpisodes


        agent = FCHC(theta, sigma, evaluateFunction, numEpisodes)
        
        count = 0

        for iteration in range(numIterations):
        
            print("Trial: %d" % (trial, ))
            print("Iteration: %d" % (iteration, ))
            p = agent.train()
            print(returns[trial][iteration * numEpisodes : count])
            print(np.mean(returns[trial][iteration * numEpisodes : count]))
            l = [[0 for i in range(5)] for j in range(5)] 
            for i in range(25):
                k = tabular_softmax.getActionProbabilities(i)
                print(k)
                r = np.argmax(k)
                if(r == 0):
                    l[i//5][i % 5] = '↑'
                elif(r == 1):
                    l[i//5][i % 5] = '↓'
                elif(r == 2):
                    l[i//5][i % 5] = '←'
                elif(r == 3):
                    l[i//5][i % 5] = '→'

            for i in range(5):
                print(l[i])
        print(p)
            
    plot(returns, 'More-Watery 687-Gridworld domain First Choice Hill Climbing (standard deviation error bars) - 50 trials', 3)



def problem3():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this assignment) on the More-Watery 687-Gridworld domain. Report the same 
    quantities.
    """

    #TODO
    
    populationSize = 40 # 40
    numElite = 20 # 20
    numEpisodes = 20 # 20
    numTrials = 50 #50
    numIterations = 100 # 100
    Kp = 30 # 30
    alpha = 3.0 # 3.0

    returns = np.zeros((numTrials, numEpisodes * numIterations * populationSize))
    
    for trial in range(numTrials):
        

        np.random.seed(np.random.randint(10000))
    
        gridworld = Gridworld()

        tabular_softmax = TabularSoftmax(25, 4)
        theta = np.random.randn(tabular_softmax.parameters.shape[0])
        
        count = 0


        def evaluateFunction(theta, numEpisodes):
            nonlocal count

            expected_reward = 0

            numTimeSteps = 10000
            tabular_softmax.parameters = theta

            for episode in range(numEpisodes):
                state = gridworld.state
                G = 0
                discount = 1
                for t in range(numTimeSteps):
                    action = tabular_softmax.samplAction(state);
                    nextstate, reward, end = gridworld.step(action)
                    G += (discount) * reward
                    discount *= gridworld.gamma
                    if end == True:
                        break
                    elif t == 200:
                        G = -50
                        break
                    state = nextstate
                expected_reward += G
                returns[trial][count] = G
                gridworld.reset()
                count += 1

            return expected_reward / numEpisodes
        
        def initPopulation(populationSize : int) -> np.ndarray:
            return np.random.randn(populationSize, tabular_softmax.parameters.shape[0])


        agent = GA(populationSize, evaluateFunction, initPopulation, numElite, numEpisodes, Kp, alpha)

        for iteration in range(numIterations):
        
            print("Trial: %d" % (trial, ))
            print("Iteration: %d" % (iteration, ))
            p = agent.train()
            print(returns[trial][iteration * numEpisodes * populationSize : count])
            print(np.mean(returns[trial][iteration * numEpisodes * populationSize : count]))
            l = [[0 for i in range(5)] for j in range(5)] 
            for i in range(25):
                k = tabular_softmax.getActionProbabilities(i)
#                 print(k)
                r = np.argmax(k)
                if(r == 0):
                    l[i//5][i % 5] = '↑'
                elif(r == 1):
                    l[i//5][i % 5] = '↓'
                elif(r == 2):
                    l[i//5][i % 5] = '←'
                elif(r == 3):
                    l[i//5][i % 5] = '→'

            for i in range(5):
                print(l[i])
        print(p)
            
    plot(returns, 'More-Watery 687-Gridworld domain Genetic Algorithm (standard deviation error bars) - 50 trials', 3)


def problem4():
    """
    Repeat the previous question, but using the cross-entropy method on the 
    cart-pole domain. Notice that the state is not discrete, and so you cannot 
    directly apply a tabular softmax policy. It is up to you to create a 
    representation for the policy for this problem. Consider using the softmax 
    action selection using linear function approximation as described in the notes. 
    Report the same quantities, as well as how you parameterized the policy. 
    
    """

    #TODO
    
    popSize = 10 #10
    numElite = 5 #5
    epsilon = 4.0 #4.0
    sigma = 1.0 #1.0
    numEpisodes = 20 #20
    numTrials = 5 #5
    numIterations = 40 #40
    k = 2 #2

    returns = np.zeros((numTrials, numEpisodes * numIterations * popSize))
    
    for trial in range(numTrials):
        

        np.random.seed(np.random.randint(10000))
    
        cartpole = Cartpole()

        tabular_softmax = TabularSoftmaxContinuous(k, 2)
        theta = np.random.randn(tabular_softmax.parameters.shape[0])
        
        count = 0


        def evaluateFunction(theta, numEpisodes):
            nonlocal count

            expected_reward = 0

            numTimeSteps = 1000
            tabular_softmax.parameters = theta

            for episode in range(numEpisodes):
                state = cartpole.state
                G = 0
                discount = 1
                for t in range(numTimeSteps):
                    action = tabular_softmax.samplAction(state);
                    nextstate, reward, end = cartpole.step(action)
                    G += (discount) * reward
                    discount *= cartpole.gamma
                    if end == True:
                        break
                    state = nextstate
                expected_reward += G
                returns[trial][count] = G
                cartpole.reset()
                count += 1

            return expected_reward / numEpisodes


        agent = CEM(theta, sigma, popSize, numElite, numEpisodes, evaluateFunction, epsilon)

        

        for iteration in range(numIterations):
        
            print("Trial: %d" % (trial, ))
            print("Iteration: %d" % (iteration, ))
            p = agent.train()
            print(returns[trial][iteration * numEpisodes * popSize : count])
#             l = [[0 for i in range(5)] for j in range(5)] 
#             for i in range(25):
#                 s = tabular_softmax.getActionProbabilities(i)
#                 print(s)
#                 r = np.argmax(s)
#                 if(r == 0):
#                     l[i//5][i % 5] = '↑'
#                 elif(r == 1):
#                     l[i//5][i % 5] = '↓'
#                 elif(r == 2):
#                     l[i//5][i % 5] = '←'
#                 elif(r == 3):
#                     l[i//5][i % 5] = '→'

#             for i in range(5):
#                 print(l[i])
        print(p)
            
    plot(returns, 'Cartpole domain Cross Entropy Method (standard deviation error bars) - 5 trials', 1000)

def problem5():
    """
    Repeat the previous question, but using first-choice hill-climbing (as 
    described in class) on the cart-pole domain. Report the same quantities 
    and how the policy was parameterized. 
    
    """
    #TODO
    
    
    sigma = 1.0

    numEpisodes = 150 #100
    numTrials = 50 #50
    numIterations = 75 #50
    k = 2 #2

    returns = np.zeros((numTrials, numEpisodes * numIterations))
    
    for trial in range(numTrials):
        

        np.random.seed(np.random.randint(10000))
    
        cartpole = Cartpole()

        tabular_softmax = TabularSoftmaxContinuous(k, 2)
        theta = np.random.randn(tabular_softmax.parameters.shape[0])
        
        count = -1


        def evaluateFunction(theta, numEpisodes):
            nonlocal count

            expected_reward = 0

            numTimeSteps = 1000
            tabular_softmax.parameters = theta

            for episode in range(numEpisodes):
                state = cartpole.state
                G = 0
                discount = 1
                for t in range(numTimeSteps):
                    action = tabular_softmax.samplAction(state);
                    nextstate, reward, end = cartpole.step(action)
                    G += (discount) * reward
                    discount *= cartpole.gamma
                    if end == True:
                        break
                    state = nextstate
                expected_reward += G
                if(count != -1):
                    returns[trial][count] = G
                    count += 1
                cartpole.reset()

            return expected_reward / numEpisodes


        agent = FCHC(theta, sigma, evaluateFunction, numEpisodes)

        count = 0

        for iteration in range(numIterations):
        
            print("Trial: %d" % (trial, ))
            print("Iteration: %d" % (iteration, ))
            p = agent.train()
            print(returns[trial][iteration * numEpisodes : count])
            print(np.mean(returns[trial][iteration * numEpisodes : count]))
#             l = [[0 for i in range(5)] for j in range(5)] 
#             for i in range(25):
#                 s = tabular_softmax.getActionProbabilities(i)
#                 print(s)
#                 r = np.argmax(s)
#                 if(r == 0):
#                     l[i//5][i % 5] = '↑'
#                 elif(r == 1):
#                     l[i//5][i % 5] = '↓'
#                 elif(r == 2):
#                     l[i//5][i % 5] = '←'
#                 elif(r == 3):
#                     l[i//5][i % 5] = '→'

#             for i in range(5):
#                 print(l[i])
        print(p)
            
    plot(returns, 'Cartpole domain First Choice Hill Climbing (standard deviation error bars) - 50 trials', 1000)


def problem6():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized. 
    """
    
    #TODO
    
    populationSize = 20 #20
    numElite = 5 #5
    numEpisodes = 5 #5
    numTrials = 50 #50
    numIterations = 20 #20
    Kp = 10 #10
    alpha = 2.5 #2.5
    k = 2 #2

    returns = np.zeros((numTrials, numEpisodes * numIterations * populationSize))
    
    for trial in range(numTrials):
        

        np.random.seed(np.random.randint(10000))
    
        cartpole = Cartpole()

        tabular_softmax = TabularSoftmaxContinuous(k, 2)
        theta = np.random.randn(tabular_softmax.parameters.shape[0])
        
        count = 0


        def evaluateFunction(theta, numEpisodes):
            nonlocal count

            expected_reward = 0

            numTimeSteps = 1000
            tabular_softmax.parameters = theta

            for episode in range(numEpisodes):
                state = cartpole.state
                G = 0
                discount = 1
                for t in range(numTimeSteps):
                    action = tabular_softmax.samplAction(state);
                    nextstate, reward, end = cartpole.step(action)
                    G += (discount) * reward
                    discount *= cartpole.gamma
                    if end == True:
                        break
                    state = nextstate
                expected_reward += G
                returns[trial][count] = G
                cartpole.reset()
                count += 1

            return expected_reward / numEpisodes


        
        def initPopulation(populationSize : int) -> np.ndarray:
            return np.random.randn(populationSize, tabular_softmax.parameters.shape[0])


        agent = GA(populationSize, evaluateFunction, initPopulation, numElite, numEpisodes, Kp, alpha)

        

        for iteration in range(numIterations):
        
            print("Trial: %d" % (trial, ))
            print("Iteration: %d" % (iteration, ))
            p = agent.train()
            print(returns[trial][iteration * numEpisodes * populationSize : count])
            print(iteration * numEpisodes * populationSize)
            print(count)
#             l = [[0 for i in range(5)] for j in range(5)] 
#             for i in range(25):
#                 s = tabular_softmax.getActionProbabilities(i)
#                 print(s)
#                 r = np.argmax(s)
#                 if(r == 0):
#                     l[i//5][i % 5] = '↑'
#                 elif(r == 1):
#                     l[i//5][i % 5] = '↓'
#                 elif(r == 2):
#                     l[i//5][i % 5] = '←'
#                 elif(r == 3):
#                     l[i//5][i % 5] = '→'

#             for i in range(5):
#                 print(l[i])
        print(p)
            
    plot(returns, 'Cartpole domain Genetic Algorithm (standard deviation error bars) - 50 trials', 1000)


def main():
    
    print("hello, world")
    
    #TODO
#     problem1()
#     problem2()
#     problem3()
#     problem4()
    problem5()
#     problem6()


if __name__ == "__main__":
    main()
