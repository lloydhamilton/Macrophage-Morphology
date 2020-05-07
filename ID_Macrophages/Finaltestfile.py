from ID_Macrophages import class_detectMacrophages as Dm
classInstance = Dm.detectMacrophages('testImage2.png')
segmented = classInstance.segmentImage(0)
final_logical = classInstance.manual_classification(segmented)
print(final_logical)