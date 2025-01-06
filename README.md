![image](https://github.com/user-attachments/assets/8e371e6b-8ec7-4a67-b73f-b4989f97ac8c)# machine-learning-algorithms
My faculty thesis using Python for categorizing Greek books with the use of NLP Algorithms

#Introduction to nlp algorithms:

### Naive Bayes Algorithm  ###

The Naive Bayes algorithm is one of the most popular and efficient machine learning algorithms used for classification and prediction problems. It is a probabilistic algorithm based on Bayes' theorem. The Naive Bayes algorithm has wide applications in many fields, mainly due to its efficiency and simplicity. It is typically used in classification problems, where the goal is to categorize data into different classes or categories. The Naive Bayes classifier assumes that the presence of a specific feature in a class is independent of the presence of any other feature. This assumption simplifies calculations and makes the algorithm highly efficient.  

The Naive Bayes model is easy to construct and particularly useful for large datasets.  

*How it works  

The algorithm operates based on the formula of Bayes' theorem, which is given by:  

\[P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}\]  

**Example  

The probability \( P(A) \) represents the likelihood of an event, for example, the probability of a text containing capital letters \( P(\text{capital}) \). If we assume a text consists of 1000 words and 300 of them contain capital letters, then 700 do not. Thus, \( P(\text{capital}) = 0.30 \).  

Next, we decide to explore the probability of words that are shorter than 50 characters. We calculate:  
\[P(\text{capital} \mid \text{word} < 50 \text{ chars}) = \frac{P(\text{word} < 50 \text{ chars} \mid \text{capital}) \cdot P(\text{capital})}{P(\text{word} < 50 \text{ chars})}\]  

The term \( P(\text{word} < 50 \text{ chars}) \) is similar to \( P(\text{capital}) \) as it represents the probability of a word being capitalized. Next, we calculate the missing values. Suppose 350 out of 500 words are shorter than 50 characters, so \( P(\text{word} < 50 \text{ chars}) = 0.35 \).  

By analyzing the first paragraph of the text, we find that only 250 words have short lengths, so:  
\[P(\text{word} < 50 \text{ chars} \mid \text{capital}) = \frac{250}{300} = 0.83\]  

Thus, the result is:  
\[P(\text{capital} \mid \text{word} < 50 \text{ chars}) = \frac{0.83 \cdot 0.30}{0.35} = 0.71\]  

---

This means there is a **71% probability** that a word shorter than 50 characters contains a capital letter.


### Support Vector Machines (SVM)  ###

Support Vector Machines (SVM) are popular machine learning algorithms used for data classification and regression tasks. SVMs are highly effective in solving classification problems in two-dimensional or multi-dimensional feature spaces. The SVM algorithm is based on the concept of finding a hyperplane (or multiple hyperplanes) that best separates the training data. The hyperplane is chosen to maximize the distance between two classes (this distance is called the "margin"). The data points closest to the margin are referred to as "support vectors." This methodology makes the SVM algorithm very efficient.  

*  Applications  

SVM algorithms are primarily used in two types of problems:  
1. **Classification Problems:** SVMs separate data into two or more classes by finding a hyperplane that maximizes the margin between the classes.  
2. **Regression Problems:** SVMs predict numerical values by finding a hyperplane that minimizes the prediction error.  

### Flexibility and Versatility  

SVMs are highly flexible because they can utilize various kernels to map data, allowing the handling of non-linear relationships between features. They are widely applied in areas such as:  
- Pattern recognition  
- Image classification  
- Speech recognition  
- And many more  

** Key Mathematical Concepts  

In binary classification, the goal is to find a hyperplane that separates two sample categories with the maximum margin. The equation of the hyperplane is:  
\[ w \cdot x - b = 0 \]  
Where:  
- \( w \): The vector determining the direction of the hyperplane  
- \( x \): The feature vector of a sample  
- \( b \): A constant called bias  

SVM determines the hyperplane that maximizes the margin between the two categories. This is achieved using support vectors and solving a convex optimization problem.  

The precise form of the equation for the distance from the hyperplane and the margin computation depends on the optimization function and the conditions set for the distribution of samples.  

---  

This structured approach to SVM ensures its effectiveness in diverse machine learning tasks.

### k-Nearest Neighbors (KNN)  ###

The k-Nearest Neighbors (KNN) algorithm is a simple and effective machine learning method for classification and regression tasks. Its core principle is to find the "nearest neighbors" of a data point based on their distances in a multidimensional space. Predictions and decisions are made based on the majority class of the nearest neighbors.  

* How KNN Works  

The KNN algorithm involves the following steps:  
1. **Define the number of neighbors (k)** to be considered.  
2. **Calculate the distance** between the observed point and all the training data points.  
3. **Select the k points** with the smallest distances.  
4. **Make a prediction** based on:  
   - The majority class among the k neighbors for classification.  
   - The average of the k neighbors' values for regression.  

** Key Assumption  

KNN operates under the assumption that samples close to each other are likely to belong to the same class. The parameter \( k \) represents the number of nearest neighbors considered when classifying a new sample. During classification, the \( k \) closest samples determine the category of the new data point based on the majority vote.  

This simplicity and intuitive approach make KNN a versatile and widely used algorithm in various applications.


