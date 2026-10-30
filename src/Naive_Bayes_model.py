import numpy as np
class NaiveBayes:
    """
    Naive Bayes classifier implementation using Gaussian distribution assumption.
    """

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.

        Parameters:
        - X (numpy array): Training feature data of shape (n_samples, n_features).
        - y (numpy array): Target labels of shape (n_samples,).

        Returns:
        - None
        """
        # Get the number of samples, number of features respectively
        self.m, self.n = X.shape
        
        # Get the unique elements in y (class labels)
        self.unique_classes = np.unique(y)
        # Get the number of unique classes
        self.n_unique = len(self.unique_classes)
        # Create empty arrays to store mean, variance, and priors
        self.mean = np.zeros((self.n_unique, self.n))
        self.variance = np.zeros((self.n_unique, self.n))
        self.priors = np.zeros(self.n_unique)
        
        for i, c in enumerate(self.unique_classes):
            # Get the portion of the data where y is equal to a certain class
            X_c = X[y == c]
            # Calculate the mean for each class and all features
            self.mean[i, :] = np.mean(X_c, axis=0)
            # Calculate the variance for each class and all features
            self.variance[i, :] = np.var(X_c, axis=0)
            # Calculate the priors
            self.priors[i] = X_c.shape[0] / self.m

    def gaussian_density(self, x, c):
        """
        Calculate the Gaussian density function for a given feature vector and class.

        Parameters:
        - x (numpy array): Feature vector of shape (n_features,).
        - c (int): Index of the class.
        Returns:
        - numpy array: Gaussian density values for each feature.
        """
        # Get the mean and the variance for the specified class
        mean = self.mean[c]
        variance = self.variance[c]
        
        # Calculate the Gaussian density function
        const = 1 / np.sqrt(variance * 2 * np.pi)
        proba = np.exp(-0.5 * ((x - mean) ** 2 / variance))
        
        return const * proba

    def get_probability(self, x):
        """
        Calculate the probability of each class given a feature vector.

        Parameters:
        - x (numpy array): Feature vector of shape (n_features,).

        Returns:
        - int: Predicted class label.
        """
        # Create an empty list to store the posteriors
        posteriors = []
        
        
        for i, c in enumerate(self.unique_classes):
            # Calculate the log of the prior
            prior = np.log(self.priors[i])
            # Calculate the new posterior and append it to the list
            posterior = np.sum(np.log(self.gaussian_density(x, i)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
        # Return the class with the highest class probability
        return self.unique_classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Predict the class labels for the input feature data.

        Parameters:
        - X (numpy array): Feature data of shape (n_samples, n_features).

        Returns:
        - numpy array: Predicted class labels of shape (n_samples,).
        """
        # Create an empty array to store the predictions
        predictions = []
        
        # Loop over each sample in X
        for x in X:
            # Get the prediction for this sample
            pred = self.get_probability(x)
            # Append the prediction to the predictions list
            predictions.append(pred)
        
        return np.array(predictions)
        