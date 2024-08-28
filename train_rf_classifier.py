from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

OUTPUT_FILE = 'random_forest/rf_classifier.sav'
def main():
    # Load the Iris dataset as an example
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier on the training data
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier,OUTPUT_FILE)
    print('saved')

if __name__=="__main__":
    main()