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


### Random Forest  

The **Random Forest** algorithm is a powerful machine learning tool that combines multiple decision trees to create a "forest" of trees. Each tree independently makes decisions based on different subsets of the data, and their predictions are aggregated to produce a final output. Random Forest uses **bootstrap sampling** to randomly sample data points with replacement during the training process. This technique reduces overfitting and improves model stability.  

* How Random Forest Works  

1. **Bootstrap Sampling**: A random subset of the data is used to train each decision tree.  
2. **Feature Subset Selection**: At each split in the tree, a random subset of features is considered to determine the best split.  
3. **Voting Mechanism**:  
   - For **classification**: Each tree votes for a class, and the class with the majority votes is chosen.  
   - For **regression**: The average prediction of all trees is used as the final output.  

** Key Advantages  

- **Versatility**: Applicable to both classification and regression problems.  
- **Reduced Overfitting**: By aggregating multiple trees, Random Forest minimizes the risk of overfitting compared to a single decision tree.  
- **Ease of Implementation**: It is straightforward to use and works well with large datasets.  

---

### Logistic Regression  

**Logistic Regression** is a machine learning algorithm primarily used for binary classification problems. It predicts whether a sample belongs to one of two categories, such as "Yes" or "No," "Spam" or "Not Spam."  

* How Logistic Regression Works  

Logistic Regression models the probability of a sample belonging to a specific class using a **Sigmoid Function**, which maps predictions to a probability range between 0 and 1. The goal is to predict the likelihood that a given input belongs to the positive class (e.g., class 1).  

** Sigmoid Function  

The Sigmoid Function is defined as:  
\[S(t) = \frac{1}{1 + e^{-t}}\]  
Here, (t)  is a linear combination of the input features.  

*** Applications  :

- **Spam detection**  
- **Customer churn prediction**  
- **Medical diagnosis** (e.g., predicting if a patient has a disease or not).  

Both **Random Forest** and **Logistic Regression** are widely used due to their effectiveness and flexibility across various domains.


### Decision Tree  

A **Decision Tree** is a popular machine learning algorithm used for both regression and classification problems. It structures data into a tree format, where each **node** represents a decision based on a feature, and each **leaf** represents a classification or regression outcome.  

* How a Decision Tree Works  

1. **Root Node**: The algorithm begins at the root node and splits the data into subsets based on the best possible feature (a process called **splitting**).  
2. **Recursive Splitting**: Each subset is treated as a new node, and the splitting process continues recursively.  
3. **Stopping Criteria**: To prevent infinite growth, a **stopping criterion** is applied, such as:  
   - Maximum tree depth  
   - Minimum number of samples per node  
   - Minimum information gain  
4. **Prediction**: Once the tree is built, new data points are classified by traversing from the root to a leaf node.

** Key Features  

- **Interpretable**: Decision Trees are easy to visualize and understand.  
- **Prone to Overfitting**: If the tree grows too large, it may overfit, meaning it will perform well on training data but poorly on unseen data.  

*** Techniques to Prevent Overfitting  

- **Depth Limitation**: Restrict the maximum depth of the tree.  
- **Pruning**: Remove unnecessary branches from the tree.  
- **Minimum Split Criteria**: Set a minimum threshold for the number of samples required to split a node.  

**** Advantages  

- Simple to understand and interpret.  
- Handles both numerical and categorical data.  
- Requires little data preprocessing.  

***** Disadvantages  

- Overfitting if the tree grows too complex.  
- Sensitive to small changes in data, which can lead to different splits and structures.  

Decision Trees are widely used in various applications and form the basis for more advanced ensemble methods like **Random Forest** and **Gradient Boosted Trees**.

### Neural Networks

**Neural Networks (NNs)**, also known as **Artificial Neural Networks (ANNs)**, represent a rapidly growing field in the realm of physical sciences. Over the past few years, this area has experienced significant advancements, largely driven by progress in computer technology. With increasing interest from scientists across various disciplines, neural networks have become widely recognized and applied in many scientific circles.  

The strong interest in Neural Networks stems from their successful application in numerous fields of science and technology. For example, in **medicine**, they are used for the recognition, analysis, and diagnosis of various diseases, as well as for analyzing electrocardiograms and electroencephalograms. In **economics**, they are employed to predict currency fluctuations, evaluate corporate bonds, assess real estate, and perform other financial calculations. Additionally, ANNs are effectively applied in **robotics**, aiding in motion control, navigation, and robotic vision, as well as in autopilot systems for airplanes and fault detection, among many other areas.  

The first reference to Neural Networks appeared in **1943** in the work of W.S. McCulloch and W. Pitts, titled "A Logical Calculus of the Ideas Immanent in Nervous Activity."  

### Artificial Neural Networks  

**Definition**:  
A Neural Network is a massively parallel distributed processor composed of simple processing units with an inherent ability to store experiential knowledge and utilize it. Key similarities with the human brain include:  
- Knowledge acquisition through a learning process.  
- Knowledge storage via synaptic weights in the connections between neurons.  

### Neuron  

The **neuron** serves as the cornerstone of an Artificial Neural Network, acting as the primary unit for information processing. An Artificial Neuron has one or more inputs, either from its environment or from other neurons in the network, which take numerical values.  

Each neuron exhibits three fundamental characteristics:  
1. **A set of connections**: Each connection is characterized by a weight, denoted as x_j  or xi_j , representing the input to synapse \( j \). The weight \( w_{nj} \), multiplied by \( x_j \), where \( n \) denotes the neuron and \( j \) the synapse input.  
2. **An adder**: This component sums all input values after they are multiplied by their respective synaptic weights.  
3. **An activation function \( g(h) \)**: This function constrains the neuron's output, typically within the closed intervals [0,1] or [-1,1].
