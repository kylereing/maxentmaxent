import numpy as np
import scipy.stats as sp
from pyemd import emd
import itertools

num_samples = 1000

candidate1 = np.random.normal(size=num_samples)
candidate2 = np.random.pareto(3,size = num_samples)
#candidate3 = np.random.uniform()
candidate4 = np.random.exponential(size = num_samples)
candidate5 = np.random.lognormal(size= num_samples)
candidate6 = np.random.poisson(size = num_samples)
candidate7 = np.random.gamma(1,size=num_samples)

candidates = [candidate1,candidate2,candidate4,candidate5,candidate6,candidate7]

#TODO: get automatic comparison to work between various initial/final distributions
def test_all_comb(cand_list):
    A = list((i,j) for ((i,_),(j,_)) in itertools.combinations(enumerate([0]*len(cand_list)),2))
    for combin in A:
        init_dist = np.ceil(cand_list[combin[0]])
        final_dist = np.ceil(cand_list[combin[1]])
        EMD,KL = calc_trajectory_with_replacement(init_dist,final_dist)
        print EMD
        print KL

def calc_trajectory_with_replacement(self,initial,final):
    '''Compare the distance between two probabilities as the initial distribution evolves with time. In this case,
    time evolution corresponds to replacing some subset of samples in the intial distribution with samples from the
    final distribution. Different distances used: Relative Entropy (KL Divergence), Earth Mover Distance (Wasserstein
    distance).
    '''

    distance = np.ones((1000,1000))
    np.fill_diagonal(distance,0)
    distance /= 1000

    # range1 = np.ptp(init_dist)
    # range2 = np.ptp(final_dist)
    EMD_history = []
    KL_history = []
    last_i = 0
    t_step = 20
    init_dist = initial
    final_dist = final
    for i in range(num_samples):
        if i % t_step == 0 and i!=0:
            init_dist = np.hstack((final_dist[last_i:i],init_dist[i:]))
            KL_t = sp.entropy(final_dist,init_dist)
            EMD = emd(final_dist,init_dist,distance)
            EMD_history.append(EMD)
            KL_history.append(KL_t)
            last_i = i
    return EMD_history,KL_history

def calc_trajectory(s_sample,f_sample):
    ''' Here, time evolution corresponds to adding on samples from the final distribution without replacing existing samples
    from the initial distribution.
    '''
    candidate1 = np.random.exponential(size=s_sample)
    candidate2 = np.random.pareto(1,f_sample)
    EMD_history = []
    KL_history = []
    last_i = 0
    t_step = 20
    init_dist = np.ceil(candidate1)
    final_dist = np.ceil(candidate2)
    for i in range(num_samples):
        if i %t_step ==0 and i!=0:
            distance = np.ones((num_samples+i,num_samples+i))
            np.fill_diagonal(distance,0)
            distance /= (num_samples+i)
            init_dist = np.hstack((init_dist,final_dist[s_sample+last_i:s_sample+i]))
            f_temp = final_dist[:s_sample+i]
            KL_t = sp.entropy(f_temp,init_dist)
            EMD = emd(f_temp,init_dist,distance)
            EMD_history.append(EMD)
            KL_history.append(KL_t)
            last_i = i
    return EMD_history,KL_history

if __name__ == '__main__':
    EMD,KL = calc_trajectory(1000,2000)
    print EMD
    print KL




