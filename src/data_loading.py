import os
import h5py
import csv

# declare some variables
INPUT_FOLDER = '../data/normalized_data'
LABEL_FILE = '../data/stage1_labels.csv'

# load labels to a dictionary
labels = {}
with open(LABEL_FILE) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        labels[row['id']] = float(row['cancer'])

# load training data
patients = os.listdir(INPUT_FOLDER)
patients.sort()
if '.DS_Store' in patients:
    patients.remove('.DS_Store')
if '._.DS_Store' in patients:
    patients.remove('._.DS_Store')

# loop over patients
for i,p in enumerate(patients):
    f = h5py.File(INPUT_FOLDER + "/" + p, "r")
    data = f['data']
    label = labels[f.attrs['id']]
    f.close
    print('Now you have the data together with the label')
    print('YOUR CODE is here')
        