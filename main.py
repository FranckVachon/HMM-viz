import matplotlib.pyplot as plt
import numpy as np


class ViterbiComponent:
    def __init__(self, description=""):
        self.description = description


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
        if not len(sequenceObj) == len(emissionObj) == len(transitionObj):
            raise Exception('all 3 lists must be same length')

        self.seqObj = []
        self.emitObj = []
        self.transitObj = []

        # the data we'll actually use

        self.transitionProbs = []
        self.sequence = []
        self.emissionProbs = []
        self.resultMatrix = []
        self.img = []
        for k, (seq,em,tran) in enumerate(zip(sequenceObj, emissionObj, transitionObj)):
            # the raw objects
            self.seqObj.append(seq)
            self.emitObj.append(em)
            self.transitObj.append(tran)

            #the data we'll actually use
            self.transitionProbs.append(tran.data)
            self.sequence.append(seq.data)
            self.emissionProbs.append(em.data)
            self.resultMatrix.append(np.zeros((tran.data.shape[0], seq.data.shape[0])))

        # init the first column of viterbi - this is just the probably of emitting the observed signal in each hidden state. Need to do for all the sequences provided
        for i,matrix in enumerate(self.resultMatrix):
            matrix[:, 0] = self.emissionProbs[i][self.sequence[i][0]]

    def dynamic_viterby(self):
        for k, (seq,em,tran, results) in enumerate(zip(self.sequence, self.emissionProbs, self.transitionProbs, self.resultMatrix)):
            # for the dynamic part. Start at one because the first column is managed in __init__()
            for idx, signal in enumerate(seq[1:len(seq)]):
                # multiply transition_probs*viter[idx][:,None] and then takes the max of the each column of the resulting matrix.
                # this is equivalent to max(vk[i]*akl) in the book
                max_va = np.amax(tran + (results[:, idx][:, None]), axis=0)

                # now we have a [m,1] vector which we need to multiply by our emission probablity matrix.
                # Each element corresonds to one hidden state, and e = [n*m]
                results[:, idx + 1] = (em + max_va[:, None])[signal, :]

                # now we have the value - need to tweak them a little so they show nicely on screen
                results = np.absolute(results)
                for col in range(0, results.shape[1]):
                    results[:, col] = results[:, col] / np.sum(results[:, col])

            self.img.append(np.amax(results) - results)
        return self.resultMatrix

    def plot_results(self):
        """plots the results calculed above
        """
        for k,(img, seq_desc) in enumerate(zip(self.img, self.seqObj)):
            row_labels = range(img.shape[0])
            col_labels = range(img.shape[1])
            plt.imshow(np.absolute(img), cmap='gray', )
            plt.xticks(range(img.shape[1]), col_labels)
            plt.yticks(range(img.shape[0]), row_labels)
            plt.title(seq_desc.description)
            plt.show()

    def show_plot(self):
        self.dynamic_viterby()
        self.plot_results()


seq = [Sequence([0,0,0,0,0,0,0,0],"Test seq all 0 "),
       Sequence([1,1,1,1,1,1,1,1], "Test seq all 1 "),
       Sequence([2,2,2,2,2,2,2,2], "Test seq all 2")]
emission = [EmissionProbs([[.5,.25,.25],  [.25,.5,.25], [.25, .25, .5]],"Test emission"),
            EmissionProbs([[.5, .25, .25], [.25, .5, .25], [.25, .25, .5]], "Test emission"),
            EmissionProbs([[.5, .25, .25], [.25, .5, .25], [.25, .25, .5]], "Test emission")]
transit = [TransitionProbability([[.95,.05,.05],  [.05,.95,.05],   [.05, .05, .95]],"Transitions"),
           TransitionProbability([[.95, .05, .05], [.05, .95, .05], [.05, .05, .95]], "Transitions"),
           TransitionProbability([[.95, .05, .05], [.05, .95, .05], [.05, .05, .95]], "Transitions")]
algo = ViterbiAlgo(seq, emission, transit)
algo.show_plot()



