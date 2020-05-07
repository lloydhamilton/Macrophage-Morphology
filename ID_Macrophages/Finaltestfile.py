from ID_Macrophages import class_detectMacrophages as Dm
import pickle
import os


classInstance = Dm.detectMacrophages('testImage2.png')
segmented = classInstance.segmentImage(0)
final_logical = classInstance.manual_classification(segmented)

# Save data variable
path = os.getcwd()
savepath = path + '\\Data1'
os.mkdir(savepath)

with open((savepath + '\\TrainingLogical1'), 'wb') as f:
    pickle.dump(final_logical, f)

with open(savepath + '\\TrainingData1', 'wb') as f:
    pickle.dump(segmented, f)

# with open('Manual Classification Data', 'rb') as f:
#     final_logical = pickle.load(f)
# with open('Manual Classification Image Data', 'rb') as f:
#     imagedata = pickle.load(f)

print(final_logical)
