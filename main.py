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
        #here we assume we have no information on what's the more likely initial state - they start out equally likely
        for i,matrix in enumerate(self.resultMatrix):
            matrix[:, 0] = (1/self.transitionProbs[i].shape[0])*self.emissionProbs[i][self.sequence[i][0]]

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
        fig = plt.figure(len(self.resultMatrix))
        fig.set_facecolor(color='black')
        for k,(img, seq_desc, em_desc, tran_desc) in enumerate(zip(self.img, self.seqObj, self.emitObj, self.transitObj)):
            row_labels = range(img.shape[0])
            col_labels = range(img.shape[1])
            fig.add_subplot(4,1, k+1)
            plt.imshow(np.absolute(img), cmap='gray', vmax=1.0)
            plt.xticks(range(img.shape[1]), col_labels, color='white')
            plt.yticks(range(img.shape[0]), row_labels, color='white')
            plt.title(seq_desc.description + em_desc.description + tran_desc.description, color='white')
        plt.show()

    def show_plot(self):
        self.dynamic_viterby()
        self.plot_results()


#Building different "typical" objects so we can combine then easily

#sequences
continuous_seq_1 = Sequence([1,1,1,1,1,1,1,1],"Continuous signal (1) - ")
alternating_seq = Sequence([0,1,0,1,0,1,0,1], "Alternating signal (0-1) - ")
long_transition_seq = Sequence([2,2,2,1,1,1,0,0,0], "Long transitions signal (2-1-0) - ")
#emissions
even_emission = EmissionProbs([[.33,.33,.33],  [.33,.33,.33], [.33, .33, .33]],"Even emission (.33) - ")
single_strong_emission = EmissionProbs([[.9, .02, .08], [.04, .9, .06], [.03, .07, .9]], "Single strong signal (.9) - ")
dual_strong_emission = EmissionProbs([[.43, .47, .1], [.1, .44, .46], [.46, .1, .44]], "Dual strong signal (.45,.45) - ")
near_even_emission = EmissionProbs([[.25,.40,.35],  [.35,.25,.40], [.4, .35, .25]],"Even-ish emission (.4, .25, .35) - ")

#transitions
unlikely_transitions = TransitionProbability([[.9999,.0001,.0001],  [.0001,.9999,.0001],   [.0001, .0001, .999]],"Unlikely transitions (.999,.0001,.0001)")
very_likely_transitions = TransitionProbability([[.01, .49, .49], [.49, .01, .49], [.49, .49, .01]], "Very likely transitions (.1,.45.45)")
near_even_transitions = TransitionProbability([[.4, .3, .3], [.3, .4, .3], [.3, .3, .4]], "Near even (.3,.4,.3)")

seq = [long_transition_seq, long_transition_seq, long_transition_seq]
emission = [even_emission, single_strong_emission,dual_strong_emission]
transit = [unlikely_transitions, unlikely_transitions, unlikely_transitions]

algo_basic = ViterbiAlgo(seq, emission, transit)
#algo_basic.show_plot()

# Now we use the same sequence to study how emission probs change the picture
#sequences
seq_e = [long_transition_seq, long_transition_seq, long_transition_seq]
emission_e = [near_even_emission, near_even_emission,near_even_emission]
transit_e = [very_likely_transitions, unlikely_transitions, near_even_transitions]


algo_emissions = ViterbiAlgo(seq_e, emission_e, transit_e)
#algo_emissions.show_plot()


continuous_seq_1 = Sequence([0,0,0,0,1,1,1,1,0,0,0,0],"Long transitions (0,1,0) - ")
stark_emiss= EmissionProbs([[0.66, .33],  [.33,.66]],"Stark (.66, .33) - ")
blurry_emiss= EmissionProbs([[0.55, .45],  [.45, .55]],"Blurry (.45, .55) - ")
likely_transitions = TransitionProbability([[.01,.99],  [.99,.01]],"Likely transitions (.99,.01)")
unlikely_transitions = TransitionProbability([[.99,.01],  [.01,.99]],"Unlikely transitions (.99,.01)")

seq_debug =[continuous_seq_1, continuous_seq_1, continuous_seq_1, continuous_seq_1]
emiss_debug =[stark_emiss, blurry_emiss, stark_emiss, blurry_emiss]
transit_debug =[likely_transitions, likely_transitions, unlikely_transitions, unlikely_transitions]
algo_test = ViterbiAlgo(seq_debug, emiss_debug, transit_debug)
algo_test.show_plot()