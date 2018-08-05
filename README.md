# kaggle_minist
i use tensorflow to realize this competition. and because of VGG model, the nets has lots of 3x3 convs,and the picture is 28x28, i just use twice of max pool.

the differen paras will make lots of influences, so, you can use different paras to training nets ,and this behavior will make your result become different. 

this program also use model persistence;  but because of memory, i just saved once model CKPT file ,and if you training stop in unknow reason, you can use follows to restore your paras. and the

ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)    
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
