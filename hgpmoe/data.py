import numpy as np
import pandas as pd
import os

DATADIR='/tigress/BEE/penn-covidsub/'

FEATURE_DICT = {'ALBUMIN': ['ALBUMIN (CCH)', 'C ALBUMIN'], 
                'CREATININE': ['C CREATININE',  'CREATININE (CCH)', 'ISTAT CREATININE'],
                'URINE': ['URINE TOTAL VOLUME'],
                'LACTIC ACID': ['LACTIC ACID'],
                'PO2 ART': [ 'PO2 ART', 'APO2 POC', 'C PO2 ARTERIAL'],
                'FIO2': ['C FIO2 ARTERIAL','FIO2 POC', 'C %FIO2','%FIO2'],
                'ANION GAP': ['ANION GAP'],
                'DDIMER': ['C D-DIMER'], 
                'AO2CT': ['AO2CT'],
                'CO2': ['APCO2 POC'],
                'CARBOXYHEMOGLOBIN': ['CARBOXYHEMOGLOBIN'],
                'METHEMOGLOBIN': ['METHEMOGLOBIN', 'C METHEMOGLOBIN'],
                'CHLORIDE': ['C CHLORIDE', 'C CHLORIDE ART', 'CHLORIDE'],
                'INR': ['C INR (INTERNATIONAL NORMALIZED RATIO)'],
                'PH': ['PH ART', 'A PH POC', 'C PH ARTERIAL', 'ARTERIAL PH (CCH)'],
                'HEMOGLOBIN': [ 'C HEMOGLOBIN'],
                'POTASSIUM': ['C POTASSIUM', 'C POTASSIUM ART'],
                'SODIUM': ['C SODIUM', 'C SODIUM ART',  'C SODIUM (ABG)', 'SODIUM (CCH)'],
                'PLATELETS': ['C PLATELETS', 'PLATELET CNT (CCH)'],
                'LACTATE': ['C LACTATE DEHYDROGENAS', 'C LACTATE POC', 'LACTATE (CCH)'],
                'BILIRUBIN': ['C BILIRUBIN, TOTAL'],
               }

def load_feature(
    feature,
    n_pts_per_attribute=30,
    model_type='gpr',
    min_trajectory_len=25,
    max_trajectory_len=80,
    train_test_split=0.75,
    cohort_name='COVID - Cohort v2.csv',
    ):
    cohort = pd.read_csv(DATADIR + cohort_name)

    # Dictionary of pt_id to race

    id_to_race_dict = dict()
    id_to_icu_dict = dict()
    id_to_vent_dict = dict()
    id_to_sex_dict = dict()

    for index, row in cohort.iterrows():
        id_to_race_dict[row['pat_id']] = row['pt_race']
        id_to_icu_dict[row['pat_id']] = row['icu_any']
        id_to_vent_dict[row['pat_id']] = row['vent_any']
        id_to_sex_dict[row['pat_id']] = row['pt_sex']

    datadir='/tigress/BEE/penn-covidsub/'
    data_df = pd.read_csv(os.path.join(DATADIR, 'COVID - Labs v2.csv'))
    data_df['order_time'] = pd.to_datetime(data_df['order_time'])
    data_df = data_df.fillna(-1)
    data_df['pt_race'] = data_df.apply(lambda row: id_to_race_dict[row.pat_id], axis=1)
    data_df['pt_sex'] = data_df.apply(lambda row: id_to_sex_dict[row.pat_id], axis=1)
    data_df['vent_any'] = data_df.apply(lambda row: id_to_vent_dict[row.pat_id], axis=1)
    data_df['icu_any'] = data_df.apply(lambda row: id_to_icu_dict[row.pat_id], axis=1)
    
    feature_df = data_df[data_df.component_name.isin(FEATURE_DICT[feature])]
    feature_df = feature_df[(feature_df.ord_num_value > 0) & (feature_df.ord_num_value < 999)]

    patient_id_list_1 = []
    patient_id_list_2 = []
    patient_id_list_3 = []
    patient_id_list_4 = []

    len_train_trajectory = 10
    for pid in feature_df.pat_id.unique():
        if len(feature_df[feature_df.pat_id == pid]) > len_train_trajectory and id_to_sex_dict[pid] == 'Male':
            if id_to_race_dict[pid] == 'White':
                patient_id_list_1.append(pid)
            else:
                patient_id_list_2.append(pid)
        elif len(feature_df[feature_df.pat_id == pid]) > len_train_trajectory and id_to_sex_dict[pid] == 'Female':
            if id_to_race_dict[pid] == 'White':
                patient_id_list_3.append(pid)
            else:
                patient_id_list_4.append(pid)

    patient_id_list = patient_id_list_1[:n_pts_per_attribute] + patient_id_list_2[:n_pts_per_attribute] + patient_id_list_3[:n_pts_per_attribute]  + patient_id_list_4[:n_pts_per_attribute]

    processed_df = feature_df[feature_df.pat_id.isin(patient_id_list)]
    
    X = []
    Y = []
    
    if model_type == 'gpr' or model_type == 'hgp':
        cluster_assignments = []
    elif model_type == 'moe':
        cluster_assignments = dict()
        
    X_test = []
    Y_test = []
    index = 0

    for j, pid in enumerate(patient_id_list):
        temp = processed_df[processed_df.pat_id == pid]
        first_time = temp.order_time.min()
        temp['t'] = (temp.order_time - first_time).dt.total_seconds() / (12 * 3600)
        temp = temp.drop_duplicates(subset=['t'])
        temp = temp.sort_values(by=['t'])
        temp = temp.reset_index()

        if len(temp) > min_trajectory_len and len(temp) < max_trajectory_len:
            Xj = temp.t.to_numpy()
            Yj = temp.ord_num_value.to_numpy()

            np.random.seed(0) 
            train_idx = np.random.choice(len(Xj), int(len(Xj) * train_test_split), replace=False)
            train_idx = sorted(train_idx)    
            X.append(Xj[train_idx])
            Y.append(Yj[train_idx])
            mask = np.ones(Yj.size, dtype=bool)
            mask[train_idx] = False
            X_test.append(Xj[mask])
            Y_test.append(Yj[mask])
            
            if model_type == 'gpr' or model_type == 'hgp':
                if pid in patient_id_list_1:
                    cluster_assignments.append(0)
                elif pid in patient_id_list_2:
                    cluster_assignments.append(1)
                elif pid in patient_id_list_3:
                    cluster_assignments.append(2)
                elif pid in patient_id_list_4:
                    cluster_assignments.append(3)
            elif model_type == 'moe':
                if pid in patient_id_list_1:
                    cluster_assignments[index] = [0, 3]
                elif pid in patient_id_list_2:
                    cluster_assignments[index] = [0, 2]
                elif pid in patient_id_list_3:
                    cluster_assignments[index] = [1,  3]
                elif pid in patient_id_list_4:
                    cluster_assignments[index] = [1, 2]
            else:
                raise Exception('Model type not currently supported')
                
            index += 1
            
    Y_arr = np.concatenate(Y)
    Y_mean = Y_arr.mean()
    Y_std = Y_arr.std()
    for j in range(len(Y)):
        Y[j] = (Y[j] - Y_mean) / Y_std
    for j in range(len(Y_test)):
        Y_test[j] = (Y_test[j] - Y_mean) / Y_std
        
    return X, Y, X_test, Y_test, cluster_assignments
    
    
    
    
