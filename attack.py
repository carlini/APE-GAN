# attack.py: generate adversarial examples on APE-GAN
# Copyright (C) 2017 Nicholas Carlini, released under the MIT License

import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf
import keras

import sys
sys.path.append("../nn/nn_robust_attacks")
import l2_attack

model = "cifar"

flags = tf.app.flags
flags.DEFINE_integer("epoch", 2, "Epoch to train [2]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28 if model == "mnist" else 32, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28 if model == "mnist" else 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "mnist" if model == "mnist" else "cifar10", "The name of dataset [cifar10, mnist, imageNet]")
flags.DEFINE_integer("c_dim", 1 if model == "mnist" else 3, "Number of color channels, 1 for greyscale otherwise 3")
flags.DEFINE_string("checkpoint_dir", "checkpoint/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("adversarial_path", "./data/advData.npy", "The path of adversarial images saved with numpy")
flags.DEFINE_string("ground_truth_path", "./data/gtData.npy", "The path of clean images saved with numpy")
flags.DEFINE_string("test_path", "./data/testData.npy", "The path of test images saved with numpy. If empty, same path with adversarial_path")
flags.DEFINE_string("save_path", "./data/resAPE-GAN.npy", "The path to save the reconstructed images with numpy.")

FLAGS = flags.FLAGS

if model == "mnist":
    from mnist_cnn import make_model, get_data
else:
    from cifar10_cnn import make_model, get_data

classifier = None

def full_model(sess, xs, n="A"):
    print("LOAD WITH", n)
    
    with tf.variable_scope(n+"QQ"):
          dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            adversarial_path=FLAGS.adversarial_path,
            ground_truth_path=FLAGS.ground_truth_path,
            test_path=FLAGS.test_path,
            save_path=FLAGS.save_path,
            advin=xs,
            c_dim=FLAGS.c_dim,
          )

    if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    classified = classifier(dcgan.sampler)

    #print('print xtest')
    #print(sess.run(dcgan.sampler, {xs: x_test[:64]}))
    #print('done')

    return dcgan.sampler, classified

(x_train, y_train), (x_test, y_test) = get_data()
def main(_):
  global classifier
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  sess = keras.backend.get_session()
  print("LOAD INIT")
  with tf.variable_scope("QQ"):
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          dataset_name=FLAGS.dataset,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          adversarial_path=FLAGS.adversarial_path,
          ground_truth_path=FLAGS.ground_truth_path,
          test_path=FLAGS.test_path,
          save_path=FLAGS.save_path,
          c_dim=FLAGS.c_dim,
          )

  if not dcgan.load(FLAGS.checkpoint_dir)[0]:
      raise Exception("[!] Train a model first, then run test mode")

  if True:
      keras.backend.set_learning_phase(False)
      classifier = make_model('linear')
      classifier.load_weights("/tmp/"+model+"model")


      class Wrap:
        num_channels = 1 if model == "mnist" else 3
        num_labels = 10
        image_size = 28 if model == "mnist" else 32
        def predict(self, xs):
            #return classifier(xs)
            return full_model(sess, xs)[1]

      attack = l2_attack.CarliniL2(sess, Wrap(), targeted=True, binary_search_steps=4,
                                   initial_const=1, max_iterations=10000, batch_size=100,
                                   learning_rate=1e-2, confidence=1,
                                   boxmin=-1, boxmax=1)
      indexs = [np.where(y_test==i)[0][0] for i in range(10)]
      indexs = np.transpose([indexs]*10).flatten()
      targets = np.array([range(10)]*10).flatten()
      targets = keras.utils.to_categorical(targets, 10)
      print(indexs.shape)
      print(targets.shape)
      adv = attack.attack(x_test[indexs], targets)
      np.save("samples/adv-"+model+"-samples.npy", adv)

      print('mean distortion', np.mean(np.sum((adv-x_test[indexs])**2,axis=(1,2,3))**.5))

      restored, classified = full_model(sess, tf.constant(adv,dtype=tf.float32), "B")

      new, preds = sess.run([restored, classified],
                            feed_dict={dcgan.advInputs: x_test[indexs]})
      
      print("Original classifier accuracy when run on the adversarial examples")
      print(np.argmax(classifier.predict(adv),axis=1)==y_test[indexs])
      print("Original classifier accuracy when run on the cleaned adversarial examples ")
      print(np.argmax(classifier.predict(new),axis=1)==y_test[indexs])

      #print(np.argmax(preds,axis=1)==y_test[:64])

def show(img):
    remap = "  .*#"+"#"*100
    img = (img.flatten()+1)*1.5
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
    
if __name__ == '__main__':
  tf.app.run()
# MNIST 4.09/4.322
## CIFAR: 0.85/1.45
