import matplotlib.pyplot as plt
import numpy as np


class ViterbiComponent:
    def __init__(self, description=""):
        self.data = description


class Sequence(ViterbiComponent):
    def __init__(self, signals_list, description=""):
        super().__init__(description)
        self.data = np.array(signals_list)


class EmissionProbs(ViterbiComponent):
    def __init__(self, emissionMatrix, description=""):
        super().__init__(description)
        self.data = np.log2(np.array(emissionMatrix))


class TransitionProbability(ViterbiComponent):
    def __init__(self, transitionMatrix, description=""):
        super().__init__(description)
        self.data = np.log2(np.array(transitionMatrix))


class ViterbiAlgo:
    """Using logprobs to handle the underflow problem"""
    def __init__(self, sequenceObj, emissionObj, transitionObj):
        #the raw objects
        self.seqObj = sequenceObj
        self.emitObj = emissionObj
        self.transitObj = transitionObj

        #the data we'll actually use
        self.transitionProbs = self.transitObj.data
        self.sequence = self.seqObj.data
        self.emissionProbs = self.emitObj.data
        self.resultMatrix = np.zeros((self.transitionProbs.shape[0], self.sequence.shape[0]))

        # init the first column of viterbi - this is just the probably of emitting the observed signal in each hidden state
        self.resultMatrix[:, 0] = self.emissionProbs[self.sequence[0]]
        self.img = np.array([])     #will hold the image to plot.
        print("new:",self.resultMatrix)
    def dynamic_viterby(self):

        # for the dynamic part. Start at one because the first column is managed in __init__()
        for idx, signal in enumerate(self.sequence[1:len(self.sequence)]):

            # multiply transition_probs*viter[idx][:,None] and then takes the max of the each column of the resulting matrix.
            # this is equivalent to max(vk[i]*akl) in the book
            max_va = np.amax(self.transitionProbs + (self.resultMatrix[:, idx][:, None]), axis=0)

            # now we have a [m,1] vector which we need to multiply by our emission probablity matrix.
            # Each element corresonds to one hidden state, and e = [n*m]
            self.resultMatrix[:, idx + 1] = (self.emissionProbs + max_va[:, None])[signal, :]

            # now we have the value - need to tweak them a little so they show nicely on screen
            self.resultMatrix = np.absolute(self.resultMatrix)
            for col in range(0, self.resultMatrix.shape[1]):
                self.resultMatrix[:, col] = self.resultMatrix[:, col] / np.sum(self.resultMatrix[:, col])

        self.img = np.amax(self.resultMatrix) - self.resultMatrix
        print(self.img)
        return self.resultMatrix

    def plot_results(self):
        """plots the results calculed above
        """
        row_labels = range(self.img.shape[0])
        col_labels = range(self.img.shape[1])
        plt.imshow(np.absolute(self.img), cmap='gray', )
        plt.xticks(range(self.img.shape[1]), col_labels)
        plt.yticks(range(self.img.shape[0]), row_labels)
        plt.show()

    def show_plot(self):
        self.dynamic_viterby()
        self.plot_results()


def dynamic_viterby_logprob(emission_probs, transition_probs, viter, obs):
    """using log prob to manage undeflow problems"""

    # init the first column of viterbi - this is just the probably of emitting the observed signal in each hidden state
    viter[:,0] = emission_probs[obs[0]]
    print("odl:",viter)

    # for the dynamic part
    for idx,observation in enumerate(obs[1:len(obs)]):

        #multiply transition_probs*viter[idx][:,None] and then takes the max of the each column of the resulting matrix.
        #this is equivalent to max(vk[i]*akl) in the book
        max_va = np.amax(transition_probs+(viter[:,idx][:,None]),axis=0)

        #now we have a [m,1] vector which we need to multiply by our emission probablity matrix.
        # Each element corresonds to one hidden state, and e = [n*m]
        viter[:, idx + 1] =(emission_probs+max_va[:,None])[observation,:]

        #now we have the value - need to tweak them a little so they show nicely on screen
        viter = np.absolute(viter)
        for col in range(0,viter.shape[1]):
            viter[:,col] = viter[:,col]/np.sum(viter[:,col])

    viter =  np.amax(viter) - viter

    return viter


def viterbi(image):
    """
    """
    row_labels = range(image.shape[0])
    col_labels = range(image.shape[1])
    plt.imshow(np.absolute(image), cmap='gray',)
    plt.xticks(range(image.shape[1]), col_labels)
    plt.yticks(range(image.shape[0]), row_labels)
    plt.show()

def sanity_checks(emission_logprobs,transition_logprobs):
    #sequence_observations_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    sequence_observations_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    #sequence_observations_2 = np.array([2, 2, 2, 2, 2, 2, 2, 2])
    l_test = [sequence_observations_1]
    for seq in l_test:
        init_viter = viterbi_matrix = np.zeros((transition_logprobs.shape[0], seq.shape[0]))
        img_log = dynamic_viterby_logprob(emission_logprobs, transition_logprobs, viterbi_matrix, seq)
        viterbi(img_log)

def parterns(emission_logprobs,transition_logprobs):
    sequence_observations_0 = np.array([0, 0, 0, 2, 2, 2, 2, 1])
    sequence_observations_1 = np.array([1, 2, 0, 1, 2, 2, 1,2])
    sequence_observations_2 = np.array([2, 2, 2, 1, 1, 1, 0, 0])
    l_test = [sequence_observations_0,sequence_observations_1,sequence_observations_2]
    for seq in l_test:
        init_viter = viterbi_matrix = np.zeros((transition_logprobs.shape[0], seq.shape[0]))
        img_log = dynamic_viterby_logprob(emission_logprobs, transition_logprobs, viterbi_matrix, seq)
        viterbi(img_log)

#the e matrix - each row is one of the possible signals we can emitt
#each column is a hidden state
#so each element is the probability of emitting a given signal for a given hidden state

emission_logprobs = np.log2(np.array([[.5,.25,.25],  [.25,.5,.25], [.25, .25, .5]]))
#matrix indicating how probably it is to change from hidden state ki hidden state kj, e.g. aij

transition_logprobs = np.log2(np.array([[.95,.05,.05],  [.05,.95,.05],   [.05, .05, .95]]))
        # parterns(emission_logprobs, transition_logprobs)

sanity_checks(emission_logprobs, transition_logprobs)

seq = Sequence([1, 1, 1, 1, 1, 1, 1, 1],"Test seq")
emission = EmissionProbs([[.5,.25,.25],  [.25,.5,.25], [.25, .25, .5]],"Test emission")
transit = TransitionProbability([[.95,.05,.05],  [.05,.95,.05],   [.05, .05, .95]],"Transitions")
algo = ViterbiAlgo(seq, emission, transit)
algo.show_plot()



