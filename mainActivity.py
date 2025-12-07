# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:20:54 2025

@author:
    
    Fábio Fernandes : 2023230805
    
    Sebastián Rivera : 2025155216
        
"""

import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.stats import kstest, f_oneway, kruskal, skew, kurtosis,shapiro,friedmanchisquare
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from embeddings_extractor import acc_segmentation,resample_to_30hz_5s,load_model
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import scikit_posthocs as sp


""""Relevant variables on the project"""

device_map = {1: "Pulso esquerdo", 2: "Pulso direito",
              3: "Peito", 4: "Perna superior direita",
              5: "Perna inferior esquerda"}

module_map = {'accel_module': "Accelerometer module",
              'gyro_module': "Gyroscope module",
              'mag_module': "Magnetometer module"
              }

NUM_PARTICIPANTS = 15
NUM_DEVICES = 5

#Directory for the folder with the part0,part1 .. folders
root = r"C:\Users\Fábio Fernandes\Desktop\3º Ano\ECAC\Projeto\FORTH_TRACE_DATASET-master\FORTH_TRACE_DATASET-master"

def load_participant_data(part_id,directory): #Return data related to the part_id participant
    
    data_part = []

    for dev_id in range(1, 6):
        
        path = os.path.join(directory, f"part{part_id}", f"part{part_id}dev{dev_id}.csv")
        
        if os.path.exists(path):

            with open(path, newline='') as f:
                reader = csv.reader(f)
                
                for line in reader:
                    
                    values = [float(x) for x in line]
                    values.append(part_id)
                    data_part.append(values)
                    
    return np.array(data_part)
 

def load_all_participants(directory,n_participants = 15): #Load data from every participant
    
    columns=["device_id","acce_x","acce_y","acce_z",
             "gyro_x","gyro_y","gyro_z",
             "magne_x","magne_y","magne_z",
             "timestamp","activity_label",
             "participant_id"]

    all_frames = []
    
    for part_id in range(0,n_participants):
        
        array = load_participant_data(part_id,directory)
        df = pd.DataFrame(array,columns=columns)
     
        all_frames.append(df)
        
    data = pd.concat(all_frames,ignore_index=True)
   
    return data


def get_module(data):

    accel_module = np.sqrt(data["acce_x"]**2 + data["acce_y"]**2 + data["acce_z"]**2)
    gyro_module = np.sqrt(data["gyro_x"]**2 + data["gyro_y"]**2 + data["gyro_z"]**2)
    magne_module = np.sqrt(data["magne_x"]**2 + data["magne_y"]**2 + data["magne_z"]**2)

    data["accel_module"] = accel_module
    data["gyro_module"] = gyro_module
    data["mag_module"] = magne_module
    
    return data


def create_boxplots_by_device(data,device_map,module_map):
    

    devices_id = data["device_id"].unique() #Get devices_id
   
    for module_col, title in module_map.items():
       
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12,25)) # 1 figure -> 5 boxplots. Each figure corresponds to a module variable, and each boxplot corresponds to a specific device
        fig.suptitle(f'Boxplots per Activity for {title}',fontsize=16,y=0.90)
    
        for device_id in devices_id:
            
            device_id = int(device_id)
            ax = axes[device_id-1]
            
         
            device_data = data[data['device_id'] == device_id] #Filter data by the device_id
            activity_labels = sorted(device_data['activity_label'].unique()) #Get the activity numbers from data
            
            plot_data = []  #Filter the previous data that was filtered by device_id, but now by the activity number and get the values of interest "accel_module" or "gyro_module" or "mag_module"
            for activity in activity_labels:    
                plot_data.append(device_data[device_data['activity_label'] == activity][module_col])
    
                
            ax.boxplot(plot_data)
                
            ax.set_title(f'Device: {device_map[device_id]}')
            ax.set_ylabel('Module')
            ax.set_xticks(range(1, len(activity_labels) + 1))
            ax.set_xticklabels(int(x) for x in activity_labels)
            ax.set_xlabel('Activity number')
            
        plt.tight_layout(rect=[0, 0, 1, 0.9]) 
        plt.show()
        plt.close(fig)
        
def calculate_outliers_density_IQR(data,module_map):
    
    data_filtered = data[data["device_id"] == 2] #Filter data by right wrist aka device 2
    
    results = []
    
    for i in module_map:
        for activity in sorted(data_filtered["activity_label"].unique()):
            sub_data_filtered = data_filtered[data_filtered["activity_label"] == activity][i]
        
            q1 = np.percentile(sub_data_filtered,25)
            q3 = np.percentile(sub_data_filtered,75)
            
            IQR = q3 - q1
            
            #Bounds 
            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR
         
            #Analysis
            n_total = len(sub_data_filtered)
            n_outliers = sum((sub_data_filtered < lower_bound) | (sub_data_filtered > upper_bound))
            density = (n_outliers / n_total) * 100
            
            results.append({
                "device_id" : 2,
                "module": i,
                "activity_label": activity,
                "outlier_density_%": density
            })
            
    return pd.DataFrame(results)

    
def detect_outliers_zscore(array,k):
    
    mean = np.mean(array)
    std = np.std(array)
  
    if std == 0:
        return np.zeros_like(array, dtype=bool)
    
    z_scores = abs((array-mean) / std)
    is_outlier = z_scores > k
    

    return is_outlier

def calculate_outliers_density_zscore(data,module_map,ks=[3,3.5,4]):

    results = []
    
    device_data = data[data["device_id"] == 2]
    activities = sorted(device_data["activity_label"].unique())

    for module in module_map:
        for k in ks:
            for activity in activities:
                
                array = device_data[device_data["activity_label"] == activity][module]
                mask = detect_outliers_zscore(array, k)
                
                n_total = len(mask)
                n_outliers = np.sum(mask)
                density = (n_outliers / n_total) * 100
                

                results.append({
                    "device_id": 2,
                    "module": module,
                    "activity_label": activity,
                    "k": k,
                    "outlier_density_%": density
                })

    return pd.DataFrame(results)


def plot_outliers_zscore(data,device_map,module_map,ks = [3,3.5,4]):
    
    devices = sorted(data["device_id"].unique())
    activities = sorted(data["activity_label"].unique())
    
    
    for module in module_map:
        for k in ks:

            fig, axes = plt.subplots(nrows=len(devices), ncols=1, figsize=(12, len(devices)*4))
            fig.suptitle(f"Outliers for {module_map[module]}, k={k}", fontsize=16)
            first = True
            
            for ax, device_id in zip(axes, devices):
                device_data = data[data["device_id"] == device_id]

                for activity in activities:
                    data_filtered = device_data[device_data["activity_label"] == activity][module]
                    mask = detect_outliers_zscore(data_filtered, k)

                    x = np.full(len(data_filtered), int(activity)) #arranges the positions for each value that we gonna plot
                    
                    ax.scatter(x[~mask], data_filtered[~mask], color="blue", s=10, label="Normal" if first else "") #Add legend only on first graphic
                    ax.scatter(x[mask], data_filtered[mask], color="red", s=10, label="Outlier" if first else "")
                    first = False #Adds a single legend per figure
                    
                ax.set_title(f"Device : {device_map[device_id]}")
                ax.set_ylabel("Module")
                ax.set_xlabel("Activity")
                ax.set_xticks(activities)

            fig.legend(loc="upper right",fontsize=15)
            plt.tight_layout(rect=[0,0,1,0.95])
            plt.show()
            plt.close('all')

def manual_kmeans(X,n_clusters=3,max_iter=100,random_state=0):
    
    np.random.seed(random_state)
    
    # choosing random centroids
    indexs = np.random.choice(len(X),n_clusters,replace=False)
    centroids = X[indexs]
    
    for iteration in range(max_iter):
        
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) 
        labels = np.argmin(distances,axis=1)
        
       # if the cluster has points we calculate the mean among the points for new centroid otherwise we keep the same
        new_centroids = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                                 for j in range(n_clusters)])

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, labels
    
def apply_kmeans(data, devices, n_clusters=3, max_iter=100):
    
    results = []

    for device_id in devices:
        device_data = data[data['device_id'] == device_id]
        activities = sorted(device_data['activity_label'].unique())

        for activity in activities:
            subset = device_data[device_data['activity_label'] == activity][
                ['accel_module', 'gyro_module', 'mag_module']].to_numpy()
            

            centroids , labels = manual_kmeans(subset,n_clusters=n_clusters,max_iter=max_iter,random_state=0)
            
            distances = np.linalg.norm(subset - centroids[labels], axis=1)
            q1 = np.percentile(distances,25)
            q3 = np.percentile(distances,75)
            IQR = q3 - q1
            
            threshold = q3 + 1.5 * IQR
            mask_outlier = distances > threshold
            
            results.append({
                'device_id': device_id,
                'activity_label': activity,
                'centroids': centroids,
                'labels': labels,
                'mask_outlier': mask_outlier,
                'data': subset,
                'outlier_density': (sum(mask_outlier) / len(mask_outlier)) * 100
                })

    return results

def plot_kmeans_results(results,devices):
    
    for result in results:
        
        device_id = result['device_id']
        
        if device_id in devices:
            
        
            activity = result['activity_label']
            subset = result['data']
            labels = result['labels']
            centroids = result['centroids']
            mask_outlier = result['mask_outlier']
            n_clusters = len(centroids)
            
             
            inliers = subset[~mask_outlier]
            inlier_labels = labels[~mask_outlier]
            outliers = subset[mask_outlier]
            
      
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
          
            ax.scatter3D(
                inliers[:, 0], inliers[:, 1], inliers[:, 2],
                c=inlier_labels, 
                cmap='viridis',
                s=40, 
                alpha=0.7,
                label='Inliers (por Cluster)'
            )
            
            ax.scatter3D(
                outliers[:, 0], outliers[:, 1], outliers[:, 2],
                c='red',
                marker='s',
                s=80,
                edgecolor='k',
                label='Outliers (Dist > Limiar)'
            )
            
            ax.scatter3D(
                centroids[:, 0], centroids[:, 1], centroids[:, 2],
                c='black',
                marker='X',
                s=200,
                edgecolor='white',
                label=f'Centróides (k={n_clusters})'
            )
            

            ax.set_xlabel('Accel Module')
            ax.set_ylabel('Gyro Module')
            ax.set_zlabel('Mag Module')
            title = f'K-Means: Dispositivo {device_id} - Atividade {activity}'
            ax.set_title(title)
            ax.legend()
        
        
            plt.show()


def apply_dbscan(data, devices, eps=0.3, min_samples=10):
    
    """
    taking too much memory and destroyed my pc
    
    """
    
    results = []

    for device_id in devices:
        device_data = data[data['device_id'] == device_id]
        activities = sorted(device_data['activity_label'].unique())

        for activity in activities:
            subset = device_data[device_data['activity_label'] == activity][
                ['accel_module', 'gyro_module', 'mag_module']
            ].to_numpy()

            X_scaled = StandardScaler().fit_transform(subset)

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            results.append({
                'device_id': device_id,
                'activity_label': activity,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'data': X_scaled
            })

    return results

def use_kolmogorov(data,modules):
    
    results = {}

    for mod in modules:
        results[mod] = {}
     
        normal_flags = []
        
        activities = data["activity_label"].unique()
        
        normal_flags = []
        activity_values = []
        for act in activities:
            
            vals = data[data['activity_label']==act][mod]
            activity_values.append(vals.values)
            
            # KS test
            stat, p = kstest(vals, 'norm', args=(vals.mean(), vals.std()))
            normal = p > 0.05
            normal_flags.append(normal) #normal if p_value > 0.05
            
        # Statistical tests
        if all(normal_flags): #-> ANOVA
            test_stat, test_p = f_oneway(*activity_values)
            test_name = "ANOVA"
        else:
            #if any of the activities not normal → Kruskal-Wallis
            test_stat, test_p = kruskal(*activity_values)
            test_name = "Kruskal-Wallis"

        results[mod]['test'] = test_name
        results[mod]['stat'] = test_stat
        results[mod]['p_value'] = test_p

    return results
 
def sliding_features(data,module_map):
    
    window_size_ms = 5000    
    step = window_size_ms // 2      
    participants = sorted(data['participant_id'].unique())
    
    features = []
    
 
    for part in participants:
        
        data_part = data[data['participant_id'] == part]
        end_time = int(np.ceil(np.max(data_part['timestamp'])))
        
        for start in range(0,end_time,step):
        
            end = start + window_size_ms
    
            mask_time = ((data_part['timestamp'] >= start) & (data_part['timestamp'] < end))
            w_values = data_part[mask_time]
        
      
            w_activities = w_values['activity_label']
            w_activities_subfiltered = np.unique(w_activities)
        
            if len(w_activities_subfiltered) == 1:  
                
                w_act_filtered = np.max(w_activities)
                
                feature_row = {
                               'activity': w_act_filtered,
                               'start':start,
                               'end':end,
                               'participant_id':part
                               }
                
                axes = ['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z', 'magne_x', 'magne_y', 'magne_z']
                for axis in axes:
                    
                    sig = w_values[axis].values
                    
                    feature_row[f'{axis}_mean'] = np.mean(sig)
                    feature_row[f'{axis}_median'] = np.median(sig)
                    feature_row[f'{axis}_std']  = np.std(sig)
                    feature_row[f'{axis}_var'] = np.var(sig)
                    feature_row[f'{axis}_rms']  = np.sqrt(np.mean(sig**2))
                    feature_row[f'{axis}_avg_der'] = np.mean(np.abs(np.diff(sig)))
                    feature_row[f'{axis}_skew'] = skew(sig)
                    feature_row[f'{axis}_kurt'] = kurtosis(sig)
                    
                    
                    q1 = np.percentile(sig,25)
                    q3 = np.percentile(sig,75)
                    feature_row[f'{axis}_IQR'] = q3 - q1
                    
                    # Mean Crossing Rate (MCR)
                    mean_crossings = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0)
                    feature_row[f'{axis}_MCR'] = mean_crossings / len(sig)
                    
                    # Zero crossing rate
                    zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
                    feature_row[f'{axis}_ZCR'] = zero_crossings / len(sig)
                        
                
                # Features espectrais
                    
                    feature_row[f'{axis}_energy'] = energy(sig)
                    feature_row[f"{axis}_DF"] = dominant_frequency(sig)
                    
                
                ax = w_values['acce_x'].values
                ay = w_values['acce_y'].values
                az = w_values['acce_z'].values   
                
                MI_accel = np.sqrt(ax**2 + ay**2 + az**2)
                feature_row['AI_accel'] = np.mean(MI_accel)
                feature_row['VI_accel'] = np.var(MI_accel)
                
                #CAGH
                feature_row['CAGH_accel'] = calculate_CAGH(ax, ay, az)
                
                SMA = (np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / len(w_values)
                feature_row['SMA'] = SMA
                
                
                matrix = np.vstack([ax, ay, az])
                cov_matrix = np.cov(matrix)
                eigenvalues = np.linalg.eigvals(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  
                feature_row['EVA1'] = eigenvalues[0]
                feature_row['EVA2'] = eigenvalues[1]
                
                
                feature_row['AAE'] = (np.mean(ax**2) + np.mean(ay**2) + np.mean(az**2)) / 3

                gx = w_values['gyro_x'].values
                gy = w_values['gyro_y'].values
                gz = w_values['gyro_z'].values
                feature_row['ARE'] = (np.mean(gx**2) + np.mean(gy**2) + np.mean(gz**2)) / 3
                
                
                features.append(feature_row)
             
    
    df_features = pd.DataFrame(features)
    df_features = df_features.reset_index(drop=True)
    
    return df_features

def calculate_CAGH(ax, ay, az):
    
    # heading direction
    heading_mag = np.sqrt(ay**2 + az**2)

    # correlation between gravity and heading
    cagh = np.corrcoef(ax, heading_mag)[0, 1]

    return cagh

def energy(signal):
    
    N = len(signal)
    
    fft_vals = np.fft.fft(signal)
    signal_magnitude = np.abs(fft_vals) ** 2
    signal_magnitude = signal_magnitude[1:]
    
    energy = np.sum(signal_magnitude)
    energy = energy / N
    
    return energy

def dominant_frequency(signal):

    N = len(signal)
    
    fs = 51.5
    
    fft_vals = np.fft.fft(signal)
    power = np.abs(fft_vals) ** 2
    
    
    freqs = np.fft.fftfreq(N, d=1/fs)
    positive_freqs = freqs[:N//2]
    positive_power = power[:N//2]
    

    positive_power = positive_power[1:]
    positive_freqs = positive_freqs[1:]
    
    
    max_idx = np.argmax(positive_power)
    dominant_freq = positive_freqs[max_idx]
    
    return dominant_freq

def pca(features):
    
    X = features.drop(columns=['activity','start','end'])    #Drop non useful data for PCA
    
    mean = X.mean()
    std = X.std()
   
    X_scaled = (X - mean) / std
    
    pca=PCA()
    pca.fit(X_scaled)

    pca_acum = np.cumsum(pca.explained_variance_ratio_)
 
    return pca_acum
    
def fisher_score(features):
    
    col_remove = ['activity','end', 'start','participant_id'] #non features
    
    X = features.drop(columns=col_remove) #drop them
    features_names = X.columns
    X = X.values #get values from columns
    
    y = features['activity'].values #get activities

    classes = np.unique(y)         #get unique activities
    n_features = X.shape[1]        #number of features
    scores = np.zeros(n_features)  #creat the scores 
    
    for i in range(n_features):
        
        numerator , denominator = 0,0
        
        f = X[:, i]   #values from each feature
        mean_i = np.mean(f) #global mean feature
        
        for c in classes:
            
            f_c = f[y == c]  #values of feature i  for class c ( certain activity)
            n_c = len(f_c) # number of segments in class
            mean_c = np.mean(f_c) # mean on the class
            var_c = np.var(f_c) # variance of class
            
            
            numerator += n_c * ((mean_c - mean_i) ** 2)
            denominator += n_c * var_c
        
        scores[i] = numerator / denominator
    
    return pd.DataFrame({ 'feature': features_names,
                         'fisher_score': scores
                        }).sort_values('fisher_score', ascending=False)

def relieff(features):
    
    col_remove = ['activity','end', 'start','participant_id']
    
    X = features.drop(columns=col_remove)
    y = features['activity'].values

    relief = ReliefF(n_neighbors=100, n_features_to_select=X.shape[1])
    
    relief.fit(X.values, y)
    
    scores = relief.feature_importances_
    
    df_relief = pd.DataFrame({ 'feature': X.columns,
                              'reliefF_score': scores
                            }).sort_values(by='reliefF_score', ascending=False)
    
    return df_relief


# =========================================== META 2 ====================================================



def plot_balance_activities(features):
    
    activity_counts = features['activity'].value_counts().sort_index()
    
    plt.bar(activity_counts.index, activity_counts.values, color="lightgreen", edgecolor="black")
    
    plt.title("Number of segments per activity")
    plt.xlabel("Activity")
    plt.ylabel("Count")
    plt.xticks(activity_counts.index)
    plt.show()

def smote_generate(features,activity,participant_id,k,n_neighbors):
    
    """
    Generates K new SMOTE synthetic samples for activity A only for participant P.
    """
    
    #filter the segments to get only the ones from the P participant
    df_aux = features[features['participant_id'] == participant_id]
    
    # drop useless data
    drop_cols = ['activity', 'start', 'end', 'participant_id']
    
    X = df_aux.drop(columns=drop_cols) # X has the features
    y = df_aux['activity'].values #Y has the activities
    
    
    X_A = X[y == activity].values   #Get the features from the given activities
    
    
    # find neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_A)
    
    
    synthetic_samples = []
    for j in range(k):  #creat k synthetic samples

        # choose random sample
        i = np.random.randint(0, len(X_A))
        x_i = X_A[i]
    
        # find neighbors of x_i
        neighbors = nn.kneighbors([x_i], return_distance=False)
    
        # choose random neighbor except himself
        j = np.random.choice(neighbors[0][1:])
        x_j = X_A[j]
    
        # interpolation between x_i and x_j
        multiplier = np.random.rand()
        x_new = x_i + multiplier * (x_j - x_i)
        synthetic_samples.append(x_new)
        
    df_new = pd.DataFrame(synthetic_samples, columns=X.columns)
    df_new['activity'] = activity
    df_new['participant_id'] = participant_id 
 
    # metadata for synthetic samples
    df_new['start'] = -1
    df_new['end'] = -1
 
    return df_new

def plot_smote(features,new_samples,part):
    
    df_p3 = features[features['participant_id'] == 3]

    feature_cols = df_p3.drop(columns=['activity','start','end','participant_id']).columns[:2]
    
    f1 = feature_cols[0]
    f2 = feature_cols[1]
    
    plt.figure(figsize=(9,7))
    
    #scatter the real features
    plt.scatter(df_p3[f1], df_p3[f2],
                c=df_p3['activity'], cmap='tab10', alpha=0.65, label="Real samples")
    
    # scatter synthetic features
    plt.scatter(new_samples[f1], new_samples[f2],
                edgecolor='black', s=180, marker='*', color='red',
                label='Synthetic (SMOTE)')
    
    plt.title(f"Participant {part} – real samples vs SMOTE synthetic samples")
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend()
    plt.show()

def extract_embeddings_dataset_v2(data):
    
    feature_encoder = load_model()
    df = data  
    
    embeddings_list = []
    all_activities = []
    all_participants = []
    all_starts = []
    all_ends = []
    
    for p_id in range(NUM_PARTICIPANTS):
                        
        data = df[df['participant_id'] == p_id]
        

        # 2. Segmentation
        original_segments, activities , starts , ends = acc_segmentation(data)
        all_starts += starts 
        all_ends += ends
        
        # 3. Resample
        resampled_segments = [resample_to_30hz_5s(segment, 51.5)[0] 
                              for segment in original_segments]
      
        
        # 4. reshape segments to [n_segments, dimensions(xyz), time]
        x_all = np.transpose(np.array(resampled_segments), (0, 2, 1))
        print(x_all.shape)
        
        # 5. Extract the Embeddings
        # iterate over the resampled segments and pass them 
        #  through the model in batches to get the embeddings
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, x_all.shape[0], batch_size):
                xb = torch.from_numpy(x_all[i:i+batch_size]).float().to("cpu")
                # eb is the embedding
                eb = feature_encoder(xb) 
                eb = torch.mean(eb, dim=2) 
                
                for k,row in enumerate(eb.cpu().numpy()):     # percorre cada embedding do batch
                    embeddings_list.append(row)
                    all_activities.append(activities[i + k])         
                    all_participants.append(p_id)


    # [n_segments, n_embeddings + 2 (label)]
    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings["activity"] = all_activities
    df_embeddings["participant_id"] = all_participants
    df_embeddings["start"] = all_starts
    df_embeddings["end"] = all_ends
    
    print(f"EMBEDDINGS DATASET pronto. Shape: {df_embeddings.shape}")
    return df_embeddings

def extract_embeddings_dataset_v2_adapted(window_256x9):
    """
    Extract embeddings for deployment
    """
    
    import torch
    import numpy as np
    import pandas as pd

    feature_encoder = load_model()
    
    
    if isinstance(window_256x9, pd.DataFrame):
        window_256x9 = window_256x9.to_numpy()
    
    # acce_xyz
    acc_xyz = window_256x9[:, :3]  # shape (256,3)
    

    resampled = resample_to_30hz_5s(acc_xyz, 51.5)[0]  # shape (resample_time, 3)
    
    # shape [n_segments, axes, time]
    x_all = np.transpose(resampled[np.newaxis, :, :], (0, 2, 1))  # shape (1, 3, T)
    
    # embeddings
    with torch.no_grad():
        xb = torch.from_numpy(x_all).float().to("cpu")
        eb = feature_encoder(xb)  
        eb = torch.mean(eb, dim=2)  
        embeddings = eb.cpu().numpy()  
    
    
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings["activity"] = 0
    df_embeddings["participant_id"] = 0
    df_embeddings["start"] = 0
    df_embeddings["end"] = 256
    
    return df_embeddings

def check_if_equal(features,embeddings):
    
    if len(features) != len(embeddings):
        print("features and embeddings have a different number of segments!")
        print(f"features: {len(features)}, embeddings: {len(embeddings)}")
        
        return False
    
    if (not all(features["start"] == embeddings["start"]) or
        not all(features["end"] == embeddings["end"]) or 
        not all(features["activity"] == embeddings["activity"]) or
        not all(features["participant_id"] == embeddings["participant_id"])):
        
        print("Something not equal between features and embeddings!")
        return False
        
    return True
        
def split_within_participant(data,seed):
    
    train_list = []
    val_list   = []
    test_list  = []
    
    for part_id in data["participant_id"].unique():
        df_part = data[data['participant_id'] == part_id]
        
        
        # 60% train, 40% temporary
        try:
            train, temp = train_test_split(
                df_part,
                test_size=0.4,
                shuffle=True,
                stratify=df_part['activity'],     # garantees "equal" distribuition of activities on the 3 sets
                random_state=seed
            )
        except:
            train, temp = train_test_split(
                df_part,
                test_size=0.4,
                shuffle=True,
                random_state=seed
            )
            
        
        # 20% val + 20% test (50/50 temp)
        try:
            val, test = train_test_split(
                temp,
                test_size=0.5,
                shuffle=True,
                stratify=temp['activity'],
                random_state=seed
            )
        except:
            val, test = train_test_split(
                temp,
                test_size=0.5,
                shuffle=True,
                random_state=seed
            )
        
        
        
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
        
        train_df = pd.concat(train_list, ignore_index=True)
        val_df   = pd.concat(val_list, ignore_index=True)
        test_df  = pd.concat(test_list, ignore_index=True)
        
        #get the data that identifies a segment , so we can apply on embeddings and have the same separation
        cols = ['participant_id','start','end']
        keys_train = train_df[cols]
        keys_val   = val_df[cols]
        keys_test  = test_df[cols]

    return train_df, val_df, test_df , keys_train, keys_val, keys_test
 
def split_between_subjects(data):
    
    participant_ids = data['participant_id'].unique()
    shuffled = np.random.permutation(participant_ids)
    
    n = len(shuffled)

    n_train = int(n * 0.6)         
    n_val   = int(n * 0.2)            
    
    train_ids = shuffled[:n_train]
    val_ids   = shuffled[n_train:n_train + n_val]
    test_ids  = shuffled[n_train + n_val:]
    
    train_df = data[data["participant_id"].isin(train_ids)]
    val_df = data[data["participant_id"].isin(val_ids)]
    test_df = data[data["participant_id"].isin(test_ids)]
    
    #get the data that identifies a segment , so we can apply on embeddings and have the same separation
    cols = ['participant_id','start','end']
    keys_train = train_df[cols]
    keys_val   = val_df[cols]
    keys_test  = test_df[cols]
    
    return train_df, val_df, test_df,  keys_train, keys_val, keys_test
      
def create_all(train_df,val_df,test_df):
    
    #separate the labels
    y_train = train_df['activity']
    y_val   = val_df['activity']
    y_test  = test_df['activity']
    
    # separate the features
    cols_drop = ['participant_id', 'activity', 'start', 'end']
    X_train = train_df.drop(columns=cols_drop).to_numpy()
    X_val   = val_df.drop(columns=cols_drop).to_numpy()
    X_test  = test_df.drop(columns=cols_drop).to_numpy()
    
    # calculate mean and std on train_df to use now and later
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    stds[stds == 0] = 1
    
    #Normalize
    X_train = (X_train - means) / stds
    X_val   = (X_val   - means) / stds
    X_test  = (X_test  - means) / stds
    
    return {
       "X_train": X_train,
       "y_train": y_train,
       "X_val":   X_val,
       "y_val":   y_val,
       "X_test":  X_test,
       "y_test":  y_test,
       "extra": {
           "means": means,
           "stds": stds
           }
       }
    
def create_PCA_version(ALL_dict, variance = 0.90):

    X_train = ALL_dict["X_train"]
    y_train = ALL_dict["y_train"]
    
    X_val = ALL_dict["X_val"]
    y_val = ALL_dict["y_val"]
    
    X_test = ALL_dict["X_test"]
    y_test = ALL_dict["y_test"]

    
    #learn with train
    pca = PCA(n_components=variance)
    X_train_pca = pca.fit_transform(X_train)
    
    #apply to val and test using what was learned with train
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    return {
        "X_train": X_train_pca,
        "y_train": y_train,
        "X_val":   X_val_pca,
        "y_val":   y_val,
        "X_test":  X_test_pca,
        "y_test":  y_test,
        "extra": {
            "pca": pca,
            "means": ALL_dict["extra"]["means"],
            "stds":  ALL_dict["extra"]["stds"]
        }
    }
    
def create_relief_version(ALL_dict , n_features=15):
    
   max_samples=1000
   
   X_train = ALL_dict["X_train"]
   y_train = ALL_dict["y_train"]
   
   X_val = ALL_dict["X_val"]
   y_val = ALL_dict["y_val"]
   
   X_test = ALL_dict["X_test"]
   y_test = ALL_dict["y_test"]
   
   n_train = X_train.shape[0]

   if n_train > max_samples:
       idx = np.random.choice(n_train, max_samples, replace=False)
       X_fit = X_train[idx]
       y_fit = y_train.values[idx]  
       
   else:
       X_fit = X_train
       y_fit = y_train.values
       
   # ReliefF - learn on train
   relief = ReliefF(n_neighbors = 5,n_features_to_select=n_features,n_jobs=-1)
   relief.fit(X_fit, y_fit)

  
   selected_idx = relief.top_features_[:n_features]


   X_train_rel = X_train[:, selected_idx]
   X_val_rel   = X_val[:,   selected_idx]
   X_test_rel  = X_test[:,  selected_idx]

   return {
        "X_train": X_train_rel,
        "y_train": y_train,
        "X_val":   X_val_rel,
        "y_val":   y_val,
        "X_test":  X_test_rel,
        "y_test":  y_test,
        "extra": {
            "selected_idx": selected_idx,
            "relief": relief,
            "means": ALL_dict["extra"]["means"],
            "stds":  ALL_dict["extra"]["stds"]
        }
    }


def knn_classifier(X_train,y_train,X_test,k):
    
    y_train = np.asarray(y_train)
    y_prediction = []
    
    for x in X_test:
        
        #calculate distance between x and points on train
        distances = np.sqrt(np.sum((X_train - x)**2,axis=1))
        
        nearest_idx = np.argsort(distances)[:k]
        
        nearest_labels = y_train[nearest_idx]

    
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        y_prediction.append(most_common)
        
    return np.array(y_prediction)
 
#knn used for permormance
def knn_classifier_optimized(X_train, y_train, X_test, k):
    clf = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def evaluate_model(y_true, y_pred):
    metrics = {}
    
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return metrics

def find_best_k(X_train,y_train,X_val,y_val,k_values):
    
    results ={}
    
    for k in k_values:
        y_pred_val = knn_classifier_optimized(X_train, y_train, X_val, k)
        metrics_val = evaluate_model(y_val, y_pred_val)
        results[k] = metrics_val["accuracy"]
        
    best_k = max(results, key=results.get)
    return best_k


def evaluate_all_datasets(datasets, k_values):
    
    results_final = {}

    for name,ds in datasets.items():

        X_train = ds["X_train"]
        y_train = ds["y_train"]
        X_val   = ds["X_val"]
        y_val   = ds["y_val"]
        X_test  = ds["X_test"]
        y_test  = ds["y_test"]

        # 1 Best k
        best_k = find_best_k(X_train, y_train, X_val, y_val, k_values)

        # 2 — Train + Validation
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        # 3 — Evaluate on Test
        y_pred_test = knn_classifier_optimized(X_train_full, y_train_full, X_test, best_k)
        metrics_test = evaluate_model(y_test, y_pred_test)

        results_final[name] = {
            "best_k": best_k,
            "accuracy": metrics_test["accuracy"],
            "precision": metrics_test["precision"],
            "recall": metrics_test["recall"],
            "f1_score": metrics_test["f1_score"],
            "confusion_matrix": metrics_test["confusion_matrix"],
            "extra" : datasets[name]["extra"],
            "X_train_full": X_train_full,
            "y_train_full": y_train_full
        }

    return results_final

def create_splits(num_runs,datasets_names,k_values):


    all_runs_accuracy = {name: [] for name in datasets_names}  
    k_results_multiple_splits = {}
    datasets_splits = {}

    for run in range(num_runs):
        
        seed = np.random.randint(0, 10_000_000)

        # REPEAT SPLITS WITH NEW SEED
        
        # WITHIN PARTICIPANT
        train_withinF, val_withinF, test_withinF, keys_train, keys_val, keys_test = split_within_participant(features,seed)
        train_withinE = embeddings.merge(keys_train, on=['participant_id','start','end'])
        val_withinE =  embeddings.merge(keys_val, on=['participant_id','start','end'])
        test_withinE =  embeddings.merge(keys_test, on=['participant_id','start','end'])


        # BETWEEN SUBJECTS
        train_betweenF, val_betweenF, test_betweenF, keys_train, keys_val, keys_test = split_between_subjects(features)
        train_betweenE = embeddings.merge(keys_train, on=['participant_id','start','end'])
        val_betweenE   = embeddings.merge(keys_val,   on=['participant_id','start','end'])
        test_betweenE  = embeddings.merge(keys_test,  on=['participant_id','start','end'])

        #3.4
        datasets_splits["ALL_withinF"] = create_all(train_withinF,val_withinF,test_withinF)
        datasets_splits["ALL_withinE"] = create_all(train_withinE,val_withinE,test_withinE)
        datasets_splits["ALL_betweenF"] = create_all(train_betweenF, val_betweenF, test_betweenF)
        datasets_splits["ALL_betweenE"] = create_all(train_betweenE,val_betweenE,test_betweenE)

        datasets_splits["PCA_withinF"] = create_PCA_version(datasets_splits["ALL_withinF"])
        datasets_splits["PCA_withinE"] = create_PCA_version(datasets_splits["ALL_withinE"])
        datasets_splits["PCA_betweenF"] = create_PCA_version(datasets_splits["ALL_betweenF"])
        datasets_splits["PCA_betweenE"] = create_PCA_version(datasets_splits["ALL_betweenE"])

        
        datasets_splits["RELIEF_withinF"] = create_relief_version(datasets_splits["ALL_withinF"])
        datasets_splits["RELIEF_withinE"] = create_relief_version(datasets_splits["ALL_withinE"])
        datasets_splits["RELIEF_betweenF"] = create_relief_version(datasets_splits["ALL_betweenF"])
        datasets_splits["RELIEF_betweenE"] = create_relief_version(datasets_splits["ALL_betweenE"])


        k_results_multiple_splits[run] = evaluate_all_datasets(datasets_splits, k_values)
        
        for dataset_name,result_dict in k_results_multiple_splits[run].items():  #save accuracys for each model on each run
                all_runs_accuracy[dataset_name].append(result_dict["accuracy"])
        
        for model_name in datasets_splits:
      
            k_results_multiple_splits[run][model_name]["extra"] = datasets_splits[model_name]["extra"]

                
    return all_runs_accuracy,k_results_multiple_splits

def run_friedman(model_list,all_runs_accuracy):
    data = [all_runs_accuracy[m] for m in model_list]
    stat,p = friedmanchisquare(*data)
    
    return stat,p
    
def get_best_model(all_runs_accuracy, model_list,k_results_multiple_splits):
    
    mean_acc_dict = {model: np.mean(all_runs_accuracy[model]) for model in model_list}
    
    
    best_model = max(mean_acc_dict, key=mean_acc_dict.get)
    best_mean = mean_acc_dict[best_model]
    
    best_model_accuracy = max(all_runs_accuracy[best_model])
    
    for run,result_dict in k_results_multiple_splits.items():
        if result_dict[best_model]["accuracy"] == best_model_accuracy:
            best_model_to_use  = result_dict[best_model]
            
    return {
        "name": best_model,
        "mean_accuracy": best_mean,
    }, mean_acc_dict, best_model_to_use

    
def deploy_classification(sample_256x9,saved_model,name,module_map):
    
    mean = saved_model["extra"]["means"]
    std = saved_model["extra"]["stds"]
    
    
    X_df = pd.DataFrame(sample_256x9, columns=[
       'acce_x','acce_y','acce_z',
       'gyro_x','gyro_y','gyro_z',
       'magne_x','magne_y','magne_z'
    ])
    
    # creat fake data for sliding_features
    X_df['participant_id'] = 0
    X_df['timestamp'] = np.arange(len(X_df))
    X_df['activity_label'] = 0
    
    
    # 2) Feature extraction / embeddings
    if name.endswith("F"):
        X = sliding_features(X_df,module_map)
    else:
        X = extract_embeddings_dataset_v2_adapted(X_df)
      
    cols_to_drop = ['participant_id','activity','start','end']
    X_features = X.drop(columns=cols_to_drop)

    X_final = X_features.to_numpy()
    
    X_final = (X_final - mean) / std
    
    # 3) PCA
    if "pca" in saved_model.get("extra"):
        pca = saved_model["extra"]["pca"]
        X_final = pca.transform(X_final)
    
    # 4) Relief feature selection
    if "selected_idx" in saved_model.get("extra"):
        sel = saved_model["extra"]["selected_idx"]
        X_final = X_final[:, sel]


    # 5) KNN prediction
    X_train = saved_model["X_train_full"]
    y_train = saved_model["y_train_full"]


    prediction = knn_classifier(X_train, y_train, X_final, saved_model["best_k"])

    return prediction
    

if os.path.exists('dados.pkl'):
    data = pd.read_pickle('dados.pkl')  
    
else:
    data = load_all_participants(root)
    

data = get_module(data)

create_boxplots_by_device(data,device_map,module_map)

outliers_density_device2_IRQ = calculate_outliers_density_IQR(data,module_map)
plot_outliers_zscore(data,device_map,module_map)
outliers_density_device2_zscore = calculate_outliers_density_zscore(data, module_map)

kmeans_results = apply_kmeans(data, device_map, n_clusters=3)
devices_to_plot = [2,5]
plot_kmeans_results(kmeans_results, devices_to_plot)

kolmogorov_results = use_kolmogorov(data, module_map)

if os.path.exists('features.pkl'):
    features = pd.read_pickle('features.pkl')  
    
else:
    features = sliding_features(data,module_map)
   

pca_acum = pca(features)

fs_score = fisher_score(features)
relief_score= relieff(features)
    

top10_fisher = fs_score.head(10) 
top10_relief = relief_score.head(10) 
common = set(top10_fisher['feature']).intersection(set(top10_relief['feature']))

print("Comum Features:", common)

#dbscan_results = apply_dbscan(data,device_map)


#================================== META 2 ========================================

features = features[features['activity']<=7]
features = features.reset_index(drop=True)

plot_balance_activities(features)
new_samples = smote_generate(features,4,3,3,5)
plot_smote(features,new_samples,3)


if os.path.exists('embeddings.pkl'):
    embeddings = pd.read_pickle('embeddings.pkl')  
else:  
    embeddings = extract_embeddings_dataset_v2(data)


matrix_format_equal = check_if_equal(features,embeddings)
print(f"Features == embeddings : {matrix_format_equal}")


within_models = [
    "ALL_withinF",
    "ALL_withinE",
    "PCA_withinF",
    "PCA_withinE",
    "RELIEF_withinF",
    "RELIEF_withinE"
]

between_models = [
    "ALL_betweenF",
    "ALL_betweenE",
    "PCA_betweenF",
    "PCA_betweenE",
    "RELIEF_betweenF",
    "RELIEF_betweenE"
]

k_values = [1, 3, 7, 9, 11,18,30]


# ======================= 5.2 Final Analysis for 1 split ============================================
one_run_accuracy,k_result_one_split = create_splits(1, within_models + between_models ,k_values)

# =========================  5.3 - Statistical significance  ====================================
all_runs_accuracy,k_results_multiple_splits = create_splits(10, within_models + between_models ,k_values)


normality_results = {}

#Shows normal distribution, however the number of samples (10) is too low to be accurate. And when tried with num_runs=30, got not normal distribution
for name, scores in all_runs_accuracy.items():
    stat, p = shapiro(scores)
    normality_results[name] = p


# Within
stat_w, p_w = run_friedman(within_models, all_runs_accuracy)
print("Friedman (within): stat =", stat_w, " p =", p_w)

# Between
stat_b, p_b = run_friedman(between_models, all_runs_accuracy)
print("Friedman (between): stat =", stat_b, " p =", p_b)


def prepare_data_for_posthoc(model_list, all_runs_accuracy):
    """
    Converts the dict {model: [acc1, acc2, ...]}
    in a matrix N x k where N = number of runs and k = number of models.
    """
    
    # assumes that all lists have the same size (number of runs)
    num_runs = len(all_runs_accuracy[model_list[0]])
    
    # Creates matrix num_runs x num_models
    M = np.zeros((num_runs, len(model_list)))
    
    for j, model in enumerate(model_list):
        M[:, j] = all_runs_accuracy[model]
    
    
    return M


def run_nemenyi_posthoc(model_list, all_runs_accuracy):
    """
    Executs the post-hoc Nemenyi and returns a p-values matrix
    """
    
    M = prepare_data_for_posthoc(model_list, all_runs_accuracy)
    
    # Nemenyi test
    p_values = sp.posthoc_nemenyi_friedman(M)
    
    return p_values

#Within and between post-hoc 
nemenyi_within = run_nemenyi_posthoc(within_models, all_runs_accuracy)
nemenyi_between = run_nemenyi_posthoc(between_models, all_runs_accuracy)

nemenyi_within.index = within_models
nemenyi_within.columns = within_models

nemenyi_between.index = between_models
nemenyi_between.columns = between_models


print("===== Post-Hoc Nemenyi (Within) =====")
print(nemenyi_within)

print("\n===== Post-Hoc Nemenyi (Between) =====")
print(nemenyi_between)


#Using only the between models to get a model adapted to a realistic scenario
best_model,mean_acc_dict, best_model_to_use = get_best_model(all_runs_accuracy,between_models,k_results_multiple_splits)

print("===== Best model =====")
print(f"Best model considerating realistic scenario is {best_model}")


data.to_pickle('dados.pkl')
features.to_pickle('features.pkl')
embeddings.to_pickle("embeddings.pkl")