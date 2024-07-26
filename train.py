# Code adapted from Cascaded Point Completion by Yuan et al. (2019)

import argparse
from scripts.DatasetLoader import DatasetLoader
from scripts.model import Encoder, Decoder, Generator
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='./data/train_data')
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--val_data', default='./data/val_data')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--generator_learning_rate', default=1e-4, type=float)
parser.add_argument('--discriminator_learning_rate', default=1e-4, type=float)
parser.add_argument('--input_num_points', default=2048, type=int)
parser.add_argument('--gt_num_points', default=2048, type=int)
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()


def train(args):
    # Load datasets
    train_data = DatasetLoader(args.train_data, 'train', args.batch_size, shuffle=True)
    val_data = DatasetLoader(args.val_data, 'val', args.batch_size, shuffle=False)

    # Set up the Generator
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    generator = Generator(encoder, decoder).to(args.device)

    # Set up the Discriminator



    for epoch in range(args.epochs):
        for i, inputs in enumerate(data):














# def train(args):
#     inputs_pl = tf.placeholder(tf.float32, (args.batch_size, args.input_num_points, 3), 'inputs')
#     gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.gt_num_points, 3), 'ground_truths')
#
#     # Load the dataset (need to ensure that I can just obtain any batch so need to test)
#     dataset = DatasetLoader(args.train_data, mode=args.mode, batch_size=args.batch_size)
#
#     # Assuming there is no such learning rate decay yet
#     learning_rate_g = tf.constant(args.generator_learning_rate, name='lr_g')
#     learning_rate_d = tf.constant(args.discriminator_learning_rate, name='lr_d')
#
#     # Initialise the optimisers
#     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#         G_optimisers = tf.train.AdamOptimizer(learning_rate_g, beta1=0.9)
#         D_optimisers = tf.train.AdamOptimizer(learning_rate_d, beta1=0.5)
#
#     with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
#         features_partial_batch = model.