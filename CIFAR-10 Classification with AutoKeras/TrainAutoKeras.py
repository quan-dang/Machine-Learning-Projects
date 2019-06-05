from sklearn.metrics import classification_report
from keras.datasets import cifar10
import os
import autokeras as ak 

def main():
    # initialize the output directory
    outputPath = "output"

    # initialize the list of training times
    # that we allow autokeras to train
    trainingTimes = [
        60 * 60,     # 1 hour
        60 * 60 * 2  # 2 hour
    ] 

    # load the training and testing data
    print("Loading CIFAR-10 dataset...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    
    # scale the data into the range [0, 1]
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0

    # initialize the label names for the CIFAR-10 dataset
    labels = ["airplane", "automobile", "bird", "cat", "deer",
		"dog", "frog", "horse", "ship", "truck"]

    # loop over training times list
    for trainingTime in trainingTimes:
        print("Training model for {} seconds...".format(trainingTime))
        model = ak.ImageClassifier(verbose=True)
        model.fit(trainX, trainY, time_limit=trainingTime)
        model.final_fit(trainX, trainY, testX, testY, retrain=True)

        # evaluate the mode
        score = model.evaluate(testX, testY)
        preds = model.predict(testX)
        report = classification_report(testY, preds, target_names=labels)

        # save the report to disk
        p = os.path.sep.join(outputPath, "{}.txt".format(trainingTime))
        f = open(p, 'w')
        f.write(report)
        f.write("\nscore: {}".format(score))
        f.close()

