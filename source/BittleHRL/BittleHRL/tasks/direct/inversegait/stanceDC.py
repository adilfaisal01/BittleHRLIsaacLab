from .foot_positions import LegSeparationFootPositions
from .filters import lowpass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

def duty_cycle_compute(run,gaittype):
    DutyCycles=[]
    FTT,time=LegSeparationFootPositions(run)
    centroids=[]

    for i in range(np.shape(FTT)[0]):
        limbnumb=i
        FR=FTT[i,:,2] # choose which foot to be used by changing the first dimension
            
        time=list(time)
        Freq=len(FR)/(time[-1]-time[1]) # calculating frequency in hz

        if gaittype=='walk':
            FR=lowpass(FR,cutoff=1.9,sample_rate=Freq) #4.5 for trot, approximately 1.9 for walk
        elif gaittype=='trot':
            FR=lowpass(FR,cutoff=4.5,sample_rate=Freq)
        elif type(gaittype)!=str:
            print('error')
        scale=StandardScaler()
        FR=FR.reshape(-1,1)
        FR = scale.fit_transform(FR)  # scale first, uses the z=x-u/s, where x is the data point, u is the mean and s is the standard deviation


        FRKmeans=KMeans(n_clusters=2,random_state=42,n_init='auto')
        FRKmeans.fit(FR)
        labels = FRKmeans.labels_
        centercentroids=np.sort(FRKmeans.cluster_centers_.flatten())


        C1=centercentroids[0] # swing threshold
        C2=centercentroids[1] # stance threshold
        centroids.append([C1, C2])
        # classification of how well the clustering algorithm works (silhouette score)
        silhouette_avg = silhouette_score(FR, labels)
        # print(f"Average Silhouette Score: {silhouette_avg}")

        # plotting the clusters
        # plt.scatter(range(len(FR)), FR.flatten(), c=labels, cmap='coolwarm', s=25)
        # plt.xlabel('Sample Index')
        # plt.ylabel('Vertical Position (z in mm)')
        # plt.title('KMeans Clustering of Foot Vertical Position')
        # plt.colorbar(label='Cluster Label')
        # plt.show()

        # using the centroids of the clusters, we can find the threshold for swing and stance and determine

        # some kind of hysteresis, Bayesian filter, uncertain (down is stance, up is swing) to determine the ambiguous states between (-30,1)
        Z_threshold_quick= 0.5*(C1+C2) 
        dataarray=np.where(FR<=Z_threshold_quick,'STANCE','SWING').flatten()

        ## find the transition probabilities by assigning values to each of the strings for Bayesian filtering

        def transition_matrix(sequence):
            unique_states=list(set(dataarray))
            state_to_index={state :i for i, state in enumerate(unique_states)}
            index_to_state={i: state for state, i in state_to_index.items()}
            n = len(unique_states)
            transition_counts = np.zeros((n, n))

            # Count transitions
            for i in range(len(sequence) - 1):
                curr = state_to_index[sequence[i]]
                next_ = state_to_index[sequence[i + 1]]
                transition_counts[curr][next_] += 1

            # Normalize to get probabilities
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            transition_probs = transition_counts / row_sums

            return transition_probs, state_to_index, index_to_state


        probtrans,statetoind,_=transition_matrix(dataarray) # find the transiition matrix for bayesian filtering


        # 0: stance, 1: swing
        ##
        css=FRKmeans.fit_predict(FR) # classification of the clusters
        clusters={}
        means=[]
        stds=[]

        for cluster_id in np.unique(css):
            # selecting points in a given cluster
            cluster_pp=FR[labels==cluster_id].flatten()
            mean = np.mean(cluster_pp)
            std = np.std(cluster_pp)

            means.append(mean)
            stds.append(std)


        means=np.array(means)
        stds=np.array(stds)
        sorted_index=np.argsort(means)
        sorted_means=means[sorted_index]
        sorted_stds=stds[sorted_index]

        for ccd in sorted_index:
            if ccd==0: 
                    ns='STANCE'
            elif ccd==1:
                    ns='SWING'

            clusters[ns]={'mean': sorted_means[ccd], 'std':sorted_stds[ccd]} #0 is swing and 1 is stance so its inverted compared to transition matrix

        # print(clusters)

        ## fit the clusters into Gaussian distribution to find P(obs|state)
        states=['STANCE','SWING']

        def GaussianLikelihood(observations):
            likelihood=[]
            for state in states:
                mean=clusters[state]['mean']
                std=clusters[state]['std']
                chances=norm.pdf(observations,mean,std)
                likelihood.append(chances)
                
            likelihood=np.array(likelihood)
            return likelihood/np.sum(likelihood,axis=0)
            

        LLK=GaussianLikelihood(FR).squeeze().T


        ## Bayesian workflow: prior-->predict-->update

        def BayesianFiltering(transitionprobability,obs,likelihood):

            belx0=np.array([[0.5],[0.5]]) # prior and initial belief
            beliefs=[]
            for t in range(0,np.shape(obs)[0]):
                belbar=transitionprobability@belx0 #predict
                lLL = likelihood[t, :] #measurement
                posterior = belbar.flatten()* lLL # element-wise multiply and update
                posterior /= np.sum(posterior)      # normalize to sum to 1
                beliefs.append(posterior)
                # next prediction
                belx0=posterior.reshape(-1,1)
            return np.array(beliefs)

        dsf=BayesianFiltering(probtrans,FR,LLK) # Bayesian Filter probabilities for stance and swing respectively

        # plotting the probabilities and the z position of the foot for visual comparison
        # plt.subplot(3,1,1)
        # plt.plot(time,dsf[:,0],label='stance',color='r')
        # plt.grid()
        # plt.ylabel('probability')
        # plt.legend(bbox_to_anchor=(1.05,1))
        # plt.title(f'Limb number {limbnumb}')
        # plt.subplot(3,1,2)

        # plt.plot(time,dsf[:,1],label='swing')
        # plt.grid()
        # plt.ylabel('probability')
        # plt.legend(bbox_to_anchor=(1.05,1))
        # plt.subplot(3,1,3)
        # plt.plot(time,FR,color='k',label='normalized z values')
        # plt.ylabel('normalized z positions')
        # plt.axhline(C2, color='y',label='swing threshold')
        # plt.axhline(C1,color='g',label='stance threshold')
        # plt.grid()
        # plt.xlabel('time (seconds)')
        # plt.legend(bbox_to_anchor=(1.05,1))
        # plt.show()
        
        


        # using Bayesian confidence data to find the duty cycle for each foot

        stance=dsf[:,0]
        swing=dsf[:,1]
        dutycycle=np.sum(stance)/(np.sum(stance)+np.sum(swing))
        # print(f'duty cycle: {dutycycle: .3f}')
        info=np.array([limbnumb,dutycycle])
        DutyCycles.append(info)
    return DutyCycles,centroids,FTT,time,dsf
