import codecademylib3_seaborn
import matplotlib.pyplot as plt
#Importing dataset
from sklearn.datasets import load_breast_cancer
#Loading the data into a variable
breast_cancer_data = load_breast_cancer()
#Inspecting the data
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)
#Importing function to be used in splitting the data
from sklearn.model_selection import train_test_split
#Splitting the data into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
print(len(training_data))
print(len(training_labels))
#Importing K Neighbors Classifier to be used in classification
from sklearn.neighbors import KNeighborsClassifier
#Creating a KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
#Training the classifier on the training sets
classifier.fit(training_data, training_labels)
#Finding the score (accuracy) of the classifier
print(classifier.score(validation_data, validation_labels))
#The classifier does well when k is 3
#Changing k to find the best k depending on the score (accuracy)
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))
#Plotting the different validation accuracies against the different k's
k_list = list(range(1,101))
#print(x)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()