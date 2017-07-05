import argparse
import os,sys
import scipy.misc
import numpy as np
import caffe
from model import pix2pix
import tensorflow as tf
from utils import *
from ops import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the input and ouput image size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--sample_freq', dest='sample_freq', type = int, default= 200, help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type = int, default=1000, help='train, test')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--queue_capacity', dest='queue_capacity', type=int, default=50, help='queue capacity')
parser.add_argument('--queue_pre_load_num', dest='queue_pre_load_num', type=int, default=10, help='queue_pre_load_num')
parser.add_argument('--gpu_memory_ratio', dest = 'gpu_memeory_ratio',type = float, default = 0.7, help='the ratio of GPU memory used')
parser.add_argument('--caffe_segmentation', dest = 'caffe_segmentation', type = bool, default = False, help = 'Use caffe generate segmentation mask')

args = parser.parse_args()
image_path = '/mnt/public/zhang7/stanford_multiview/Stanford_Online_Products/chair_final/'
caffe_model_path = '/mnt/public/zhang7/caffe_psp/pspnet101_VOC2012.caffemodel'
caffe_prototxt_path = '/mnt/public/zhang7/pix2pix_multiview/test.prototxt'

def main(_):
	if args.caffe_segmentation is True:		
		caffenet = caffe.Net(caffe_prototxt_path, caffe_model_path, caffe.TEST)
		caffe.set_mode_gpu()
		caffe.set_device(0)
		caffe_chair_segmentation(caffenet,image_path)
	mean_val = np.array([145.99938098 , 132.40487166 , 121.54731091])
	image_generator = image_reader(image_path, args.image_size, args.batch_size,mean_val)

	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)
	if not os.path.exists(args.sample_dir):
		os.makedirs(args.sample_dir)
	if not os.path.exists(args.test_dir):
		os.makedirs(args.test_dir)
	img_prefix_num = len(get_file_name_counter_hashmap(image_path))
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memeory_ratio)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,intra_op_parallelism_threads=2)) as sess:
		model = pix2pix(sess, image_generator,args, image_size=args.image_size, batch_size=args.batch_size,
						output_size=args.image_size, checkpoint_dir=args.checkpoint_dir)

		if args.phase == 'train':
			model.train(img_prefix_num)
		else:
			model.test()

if __name__ == '__main__':
		tf.app.run()