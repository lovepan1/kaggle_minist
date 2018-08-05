import tensorflow as tf
from mnist import train, validation, test
import os

LEARNING_RATE_BASE = 0.0008
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.00004
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"
BATCH_SIZE = 50
MAX_STEPS = 15000
MOVING_AVERAGE_DECAY = 0.99

def weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer = tf.truncated_normal_initializer( stddev= 0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]
  
def ministInference(inputTensor, regularizer, keep_prob):  
#     with tf.variable_scope("layer1"):          

    with tf.variable_scope("layer1"):
        W_conv1 = weight_variable([3,3,1,32], regularizer)
        b_conv1 = bias_variable([32])       
        h_conv1 = tf.nn.relu(conv2d(inputTensor, W_conv1) + b_conv1)
    with tf.variable_scope("layer2"):
        W_conv1_2 = weight_variable([3,3,32,32], regularizer)
        b_conv1_2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv1_2) + b_conv1_2)
    # h_pool1 = max_pool_2x2(h_conv2)

    with tf.variable_scope("layer3"):
        W_conv2 = weight_variable([3,3,32,64], regularizer)
        b_conv2 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv2) + b_conv2)
    
    with tf.variable_scope("layer4"):  
        W_conv2_2 = weight_variable([3,3,64,64], regularizer)
        b_conv2_2 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv2_2) + b_conv2_2)
        h_pool1 = max_pool_2x2(h_conv4)
        
    with tf.variable_scope("layer6"):         
        W_conv3 = weight_variable([3,3,64,128], regularizer)
        b_conv3 = bias_variable([128])
        h_conv6 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    with tf.variable_scope("layer7"):                  
        W_conv3_2 = weight_variable([3,3,128,128], regularizer)
        b_conv3_2 = bias_variable([128])
        h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv3_2) + b_conv3_2)
    # h_pool2 = max_pool_2x2(h_conv6) 

    with tf.variable_scope("layer8"):     
        W_conv4 = weight_variable([3,3,128,256], regularizer)
        b_conv4 = bias_variable([256])  
        h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv4) + b_conv4)
        
    with tf.variable_scope("layer9"): 
        W_conv4_2 = weight_variable([3,3,256,128], regularizer)
        b_conv4_2 = bias_variable([128])  
        h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv4_2) + b_conv4_2)
        h_pool2 = max_pool_2x2(h_conv9)
    
    with tf.variable_scope("layer11"):        
        W_conv5 = weight_variable([3,3,128,64], regularizer)
        b_conv5 = bias_variable([64])
        h_conv11 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5) 
           
    with tf.variable_scope("layer12"):        
        W_conv5_2 = weight_variable([3,3,64,64], regularizer)   
        b_conv5_2 = bias_variable([64])  
        h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv5_2) + b_conv5_2)
    
    with tf.variable_scope("layer13"): 
        W_fc1 = weight_variable([7*7*64, 1024],regularizer)
        b_fc1 = bias_variable([1024])   
        h_pool2_flat = tf.reshape(h_conv12, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)        
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
    with tf.variable_scope("layer14"):        
        W_fc2 = weight_variable([1024, 10], regularizer)
        b_fc2 = bias_variable([10])     
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv

def training():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 784], name = 'x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name = 'y-input') 
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(x, [-1,28,28,1])        
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y_conv = ministInference(x_image, regularizer, keep_prob)
    global_step = tf.Variable(0, trainable=False)
    
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
 
    with tf.name_scope("loss_average"):    
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) 
#         loss = cross_entropy + regularizer       
        loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
        
    with tf.name_scope("train_step"):         
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 15000, LEARNING_RATE_DECAY, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)     
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
            

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("/path/to//log", tf.get_default_graph())
    writer.close()   
       
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(MAX_STEPS):
        batch = train.next_batch(BATCH_SIZE)
        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                                                  x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i%100 == 0:
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step= global_step)
            print("test accuracy %g"%accuracy.eval(feed_dict={
                                                      x: validation.images, y_: validation.labels, keep_prob: 1.0}))    
    f = open('prediction.csv', 'w+')
    f.write('Label\n')
    batchs = get_batchs(test, 50)
    i = 1
    for test_image in batchs:
        prediction = tf.argmax(y_conv, 1)
        test_labels = prediction.eval(feed_dict={x: test_image, keep_prob: 1.0})
        for label in test_labels:
            f.write(str(label) + '\n')
            i += 1
    f.close()

  
