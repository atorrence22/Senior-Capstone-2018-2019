# Dependencies
import glob, os
import tensorflow as tf
import pandas as pd
import numpy as np

# create seed for random_normal()
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
input_nodes = 3
output_nodes = 3
hidden_layer_nodes = 10

# Data preprocess
def data_preprocess(next):
    # tf_features = [next['BFSIZE'], next["HDRSIZE"], next['NODESTATE'], next['METADATASIZE'], next['STG_HINT']]
    # tf_features = [next['BFSIZE'], next["HDRSIZE"], next['METADATASIZE'], next['STG_HINT']]
    # tf_features = [next['BFSIZE'], next["HDRSIZE"], next['NODESTATE'], next['METADATASIZE']]
    tf_features = [next['BFSIZE'], next['HDRSIZE'], next['METADATASIZE']]
    # tf_features = [next['HDRSIZE'], next['METADATASIZE']]
    # tf_features = [next['BFSIZE'], next['METADATASIZE']]
    # tf_features = [next['BFSIZE'], next['HDRSIZE']]
    features_columns = np.array(sess.run(tf_features), dtype='float32')

    num_features = len(tf_features)
    # Normalization
    for column in range(num_features):
        column_range = features_columns[column].max() - features_columns[column].min()
        # print(column_range)
        if column_range != 0.0:
            features_columns[column] = (features_columns[column] - features_columns[column].min()) / column_range

    features = np.array([]).reshape(0, num_features)
    for i in range(batch_len):
        ele = np.zeros(num_features)
        for j in range(num_features):
            np.put(ele, j, features_columns[j][i])
        features = np.r_[features, [ele]]
    # print(len(features))

    tf_labels = next['label']
    labels = np.array(sess.run(tf_labels), dtype='float32')
    # print(labels)

    # One-hot encoding for the categories
    num_classes = 3
    targets = np.array([]).reshape(0, num_classes)

    for i in range(0, batch_len):
        ele = np.zeros(num_classes)
        np.put(ele, labels[i], 1)
        targets = np.r_[targets, [ele]]

    #     return features, targets

    #     Shuffle Data
    indices = np.random.choice(len(features), len(features), replace=False)
    X_data = features[indices]
    y_data = targets[indices]

    # X_values = features[indices]
    # y_values = targets[indices]

    return X_data, y_data


def training(X_train, y_train, start_epoch):
    # Interval / Epochs
    interval = 500
    epoch = 1500

    # Training the model...
    for i in range(1, (epoch + 1)):
        sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
        if i % interval == 0:
            print('Epoch', i + start_epoch, '|', 'Loss:',
                  sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))
            # print(sess.run(b1))

    return i + start_epoch

# get the accuracy of the model
def predict(X_test, y_test):
    correct_prediction = tf.equal(tf.argmax(final_output, 1), tf.argmax(y_target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("The accuracy of the model is: ", sess.run(accuracy, feed_dict={X_data: X_test, y_target: y_test}))


# define a neural network

# Initialize placeholders
X_data = tf.placeholder(shape=[None, input_nodes], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, output_nodes], dtype=tf.float32)

# We create a neural Network which contains 3 layers with 4, 10, 5 nodes repectively
w1 = tf.Variable(tf.random_normal(shape=[input_nodes, hidden_layer_nodes]))  # Weight of the input layer
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # Bias of the input layer
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, output_nodes]))  # Weight of the hidden layer
b2 = tf.Variable(tf.random_normal(shape=[output_nodes]))  # Bias of the hidden layer
hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))

# Loss Function
loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))

# Optimizer
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
# optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.001).minimize(loss)
# optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01).minimize(loss)
# optimizer = tf.train.FtrlOptimizer(learning_rate=0.01).minimize(loss)
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

print("A neural Network which contains 3 layers with", input_nodes, ", ", hidden_layer_nodes, ", ", output_nodes,
      " nodes repectively was created!")

path = '/Users/zhaoluyang/Downloads/Senior-Capstone-2018-2019-master/Notebooks/firstTestData/merge_data/feature_extraction'
# path = data_dir + "/merge_data/upscale"
all_files = glob.glob(
    os.path.join(path, "*.csv"))  # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_dataset = pd.concat(df_from_each_file, ignore_index=True)
# 44633877
# print(len(concatenated_dataset))

batch_len = 3000
dataset = tf.data.experimental.make_csv_dataset(all_files, batch_size=batch_len)
# print(dataset)

start_epoch = 0
print('Training the model...')
iter = dataset.make_one_shot_iterator()

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# There are total 2,157,764 rows in the combined csv files

# choose first 4000 rows as test data
next = iter.get_next()

X_test, y_test = data_preprocess(next)

# then 538*4000 = 2,152,000 rows will be used as training data
# 538
for i in range(20):
    next = iter.get_next()
    # next is a dict with key=columns names and value=column data
    X_train, y_train = data_preprocess(next)
    #     print(X_train)
    start_epoch = training(X_train, y_train, start_epoch)

predict(X_test, y_test)

print("Training finished\n")

# Prediction
# np.set_printoptions(precision=4)
# unknown = np.array([[0.0, 0.002894, 0.148097]], dtype=np.float32)
# predicted = sess.run(final_output, feed_dict={X_data: unknown})
# # model.predict(unknown)
# print("Using model to predict pool id for features: ", unknown)
# print("\nPredicted softmax vector is: ",predicted)
# Class_dict={'POOLID_-1000000': 0, 'POOLID_-9': 1, 'POOLID_-1': 2, 'POOLID_4': 3, 'POOLID_-1': 4, 'POOLID_6': 5, 'POOLID_42': 6, 'POOLID_72': 7, 'POOLID_82': 8 }
# pool_dict = {v:k for k,v in Class_dict.items()}
# print("\nPredicted pool id is: ", pool_dict[np.argmax(predicted)])

