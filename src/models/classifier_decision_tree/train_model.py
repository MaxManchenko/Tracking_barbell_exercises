import numpy as np
import pandas as pd
import joblib
import json
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from src.utils.LearningAlgorithms import ClassificationAlgorithms

config_path = "config.json"


# --------------------------------------------------------------
# Load config. settings and Build the path to the dataset
# --------------------------------------------------------------
with open(config_path, "r") as config_file:
    config = json.load(config_file)

data_path = config["data_path_03_features"]

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle(data_path)

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_train = df.drop(["participant", "category", "set"], axis=1)

X = df_train.drop("label", axis=1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)

y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
basic_features = ["acc_x", "acc_y", "acc_y", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3", "pca_4"]
time_features = [t for t in df_train.columns if "_temp_" in t]
frequency_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

print("Basic features", len(basic_features))
print("Square features", len(square_features))
print("PCA features", len(pca_features))
print("Time features", len(time_features))
print("Frequency features", len(frequency_features))
print("Cluster features", len(cluster_features))

feature_set_1 = basic_features
feature_set_2 = basic_features + square_features + pca_features
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

max_features = 5
# Since we have some kind of stochastic process here, run 3 iterations
selected_features = []
ordered_scores = []
for i in range(0, 3):
    print(f"Iteration {i}")
    sel_features, ord_scores = learner.forward_selection(
        max_features=max_features, X_train=X_train, y_train=y_train
    )
    selected_features.append(sel_features)
    ordered_scores.append(ord_scores)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(np.arange(1, max_features + 1, 1), ordered_scores[0], label="Iter 1")
ax.plot(np.arange(1, max_features + 1, 1), ordered_scores[1], label="Iter 2")
ax.plot(np.arange(1, max_features + 1, 1), ordered_scores[2], label="Iter 3")
ax.set_xticks(np.arange(1, max_features + 1, 1))
ax.set_xlabel("Number of features")
ax.set_ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

# The plot shows that it makes sense to use a set of 3 features.
# Add a 4th feature 'duration' as it's #2 in one of the selected_feature list
handpicked_features = [
    "acc_y_temp_mean_ws_5",
    "gyr_r_freq_0.0_Hz_ws_14",
    "acc_y_freq_0.0_Hz_ws_14",
    "duration",
]

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    handpicked_features,
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected features",
]

iterations = 1
score_df = pd.DataFrame()

# Train 5 models at the same time performing a grid search
## The code is from Mark Hoogendoorn and Burkhardt Funk,
## Machine Learning for the Quantified Self,Springer, 2017
for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            rf_best_params,
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        knn_best_params,
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        dt_best_params,
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        nb_model,
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# --------------------------------------------------------------
# Select the best model and evaluate results
# --------------------------------------------------------------
# The RF showed the best result, but DT showed almost the same result on
# a much smoller set of features, so I choose DT as the main algorithm
criterion = dt_best_params["criterion"]
min_samples_leaf = dt_best_params["min_samples_leaf"]

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.decision_tree(
    X_train[handpicked_features],
    y_train,
    X_test[handpicked_features],
    criterion=criterion,
    min_samples_leaf=min_samples_leaf,
    gridsearch=False,
)

accuracy = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
# participant_df = df.drop(["set", "category"], axis=1)

# X_train = participant_df.query('participant != "A"').drop("label", axis=1)
# y_train = participant_df.query('participant != "A"')["label"]

# X_test = participant_df.query('participant == "A"').drop("label", axis=1)
# y_test = participant_df.query('participant == "A"')["label"]

# X_train = X_train.drop("participant", axis=1)
# X_test = X_test.drop("participant", axis=1)

# fig, ax = plt.subplots(figsize=(10, 5))
# df_train["label"].value_counts().plot(
#     kind="bar", ax=ax, color="lightblue", label="Total"
# )

# y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
# y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
# plt.legend()
# plt.show()

# --------------------------------------------------------------
# Reuse the best model again and evaluate results
# --------------------------------------------------------------
# (
#     class_train_y,
#     class_test_y,
#     class_train_prob_y,
#     class_test_prob_y,
# ) = learner.decision_tree(
#     X_train[handpicked_features],
#     y_train,
#     X_test[handpicked_features],
#     criterion=criterion,
#     min_samples_leaf=min_samples_leaf,
#     gridsearch=False,
# )

# accuracy = accuracy_score(y_test, class_test_y)
# classes = class_test_prob_y.columns
# cm = confusion_matrix(y_test, class_test_y, labels=classes)

# # create confusion matrix for cm
# plt.figure(figsize=(10, 10))
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.title("Confusion matrix")
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)

# thresh = cm.max() / 2.0
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(
#         j,
#         i,
#         format(cm[i, j]),
#         horizontalalignment="center",
#         color="white" if cm[i, j] > thresh else "black",
#     )
# plt.ylabel("True label")
# plt.xlabel("Predicted label")
# plt.grid(False)
# plt.show()

# --------------------------------------------------------------
# Train the best model on the whole dataset
# --------------------------------------------------------------
data_path_processed = config["data_path_fully_processed_out"]
df = pd.read_pickle(data_path_processed)
X = df.drop("label", axis=1)
y = df["label"]

dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

dtree.fit(X, y)

# --------------------------------------------------------------
# Save the best model
# --------------------------------------------------------------
model_filename = config["model_path_classifier"]
joblib.dump(dtree, model_filename)
