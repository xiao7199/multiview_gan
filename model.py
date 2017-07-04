import tensorflow as tf
import numpy as np
from utils import *
from ops import *
import threading

class pix2pix(object):

	def __init__(self, sess, image_generator, args, image_size=256,
				 batch_size=1, sample_size=1, output_size=256,
				 gf_dim=64, df_dim=64, L1_lambda=100,
				 input_c_dim=3, output_c_dim=3, dataset_name='facades',
				checkpoint_dir=None, sample_dir=None):
		self.sess = sess
		self.batch_size = batch_size
		self.image = image_size
		self.sampe_size = sample_size
		self.output_size = output_size
		self.args = args

		self.sess = sess
		self.is_grayscale = (input_c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.sample_size = sample_size
		self.output_size = output_size

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.L1_lambda = L1_lambda
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn_e2 = batch_norm(name='g_bn_e2')
		self.g_bn_e3 = batch_norm(name='g_bn_e3')
		self.g_bn_e4 = batch_norm(name='g_bn_e4')
		self.g_bn_e5 = batch_norm(name='g_bn_e5')
		self.g_bn_e6 = batch_norm(name='g_bn_e6')
		self.g_bn_e7 = batch_norm(name='g_bn_e7')
		self.g_bn_e8 = batch_norm(name='g_bn_e8')

		self.g_bn_d1 = batch_norm(name='g_bn_d1')
		self.g_bn_d2 = batch_norm(name='g_bn_d2')
		self.g_bn_d3 = batch_norm(name='g_bn_d3')
		self.g_bn_d4 = batch_norm(name='g_bn_d4')
		self.g_bn_d5 = batch_norm(name='g_bn_d5')
		self.g_bn_d6 = batch_norm(name='g_bn_d6')
		self.g_bn_d7 = batch_norm(name='g_bn_d7')

		self.checkpoint_dir = checkpoint_dir

		self.real_data_ph = tf.placeholder(tf.float32,
										[3,self.batch_size, self.image_size, self.image_size,3],
										name='real_chairs')
		self.real_chair1_view2_mask_ph = tf.placeholder(tf.float32,
										[self.batch_size,self.image_size,self.image_size,4],
										name = 'real_chair_mask')
		self.image_generator = image_generator

		self.init_queue_op()
		self.build_model()

	def build_model(self):

		self.real_chair1_view1 = self.real_data[1,:,:,:,:];
		self.real_chair1_view2 = self.real_data[2,:,:,:,:];
		self.real_chair2 = self.real_data[3,:,:,:,:];

		self.fake_img = self.generator(self.real_chair1_view2_mask,self.real_chair1_view1)

		self.same_chair_different_view = tf.concat([self.real_chair1_view1, self.real_chair1_view2], 3)
		self.different_chair = tf.concat([self.real_chair1_view1, self.real_chair2], 3)
		self.chair_generator_compare = tf.concat([self.real_chair1_view1, self.fake_img], 3)
		
		self.D_true_same_chair_different_view, self.D_logits_true_same_chair_different_view = self.discriminator(self.same_chair_different_view, reuse=False)
		self.D_false_different_chair, self.D_logits_false_different_chair = self.discriminator(self.different_chair, reuse=True)
		self.D_false_filled_texture, self.D_logits_false_filled_testure = self.discriminator(self.chair_generator_compare, reuse=False)

		self.fake_img_sample = self.sampler(self.real_chair1_view2_mask,self.real_chair1_view1)

		self.d_same_chair = tf.summary.histogram("same_chair_different_view", D_true_same_chair_different_view)
		self.d_different_chair = tf.summary.histogram("different_chair", self.D_false_different_chair)
		self.d_filled_texture = tf.summary.histogram("filled_texture", self.D_logits_false_filled_testure)
		self.fake_B_sum = tf.summary.image("fake_image", self.fake_img)

		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_true_same_chair_different_view, 
																	labels=tf.ones_like(self.D_logits_true_same_chair_different_view)))
		self.d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_false_different_chair, 
																	labels=tf.zeros_like(self.D_logits_false_different_chair)))
		self.d_loss_fake_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_false_filled_testure, 
																	labels=tf.zeros_like(self.D_logits_false_filled_testure)))
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_false_filled_testure, labels=tf.ones_like(self.D_false_filled_texture))) \
															+ self.L1_lambda * tf.reduce_mean(tf.abs(self.real_chair2 - self.fake_img))
		
		self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum_1 = tf.summary.scalar("d_loss_fake_1", self.d_loss_fake_1)
		self.d_loss_fake_sum_2 = tf.summary.scalar("d_loss_fake_2", self.d_loss_fake_2)

		self.d_loss = self.d_loss_real + self.d_loss_fake_1 + self.d_loss_fake_2

		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

	def init_queue_op(self):
		print('[O] Start Pre-load the image to Queue' )
		self.q = tf.FIFOQueue(self.args.queue_capacity, [tf.float32, tf.float32])
		self.enqueue_op = self.q.enqueue([self.real_data_ph, self.real_chair1_view2_mask_ph])
		self.real_data, self.real_chair1_view2_mask = self.q.dequeue()
		
		for i in range(self.args.queue_pre_load_num):
			train_data = self.image_generator.next()
			self.sess.run(self.enqueue_op,feed_dict={self.real_data_ph:train_data[0],self.real_chair1_view2_mask_ph:train_data[1]})
		
		coord = tf.train.Coordinator()
		thread = tf.train.start_queue_runners(sess=self.sess, coord=coord)
		threads = []

		for _ in range(1):
		
			# This is the method that runs in the threads and feeds examples to the queue
			t = threading.Thread(target=load_and_enqueue, args=(
				self.sess,self.enqueue_op,coord,self.image_generator,self.real_data_ph,self.real_chair1_view2_mask_ph))
			t.setDaemon(True)
			t.start()
			threads.append(t)
			coord.register_thread(t)
		print('[*] Finish Pre-load the image')
	def discriminator(self, image, y=None, reuse=False):

		with tf.variable_scope("discriminator") as scope:

			# image is 256 x 256 x (input_c_dim + output_c_dim)
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False

			h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
			# h0 is (128 x 128 x self.df_dim)
			h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
			# h1 is (64 x 64 x self.df_dim*2)
			h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
			# h2 is (32x 32 x self.df_dim*4)
			h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
			# h3 is (16 x 16 x self.df_dim*8)
			h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

			return tf.nn.sigmoid(h4), h4

	def generator(self, image, real_image, y=None):

		with tf.variable_scope("generator") as scope:

			s = self.output_size
			s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

			# image is (256 x 256 x input_c_dim)
			e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
			# e1 is (128 x 128 x self.gf_dim)
			e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
			# e2 is (64 x 64 x self.gf_dim*2)
			e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
			# e3 is (32 x 32 x self.gf_dim*4)
			e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
			# e4 is (16 x 16 x self.gf_dim*8)
			e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
			# e5 is (8 x 8 x self.gf_dim*8)
			e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
			# e6 is (4 x 4 x self.gf_dim*8)
			e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
			# e7 is (2 x 2 x self.gf_dim*8)
			e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
			# e8 is (1 x 1 x self.gf_dim*8)

			# image is (256 x 256 x input_c_dim)
			e1_real = conv2d(real_image, self.gf_dim, name='g_e1_conv_real')
			# e1 is (128 x 128 x self.gf_dim)
			e2_real = self.g_bn_e2(conv2d(lrelu(e1_real), self.gf_dim*2, name='g_e2_conv_real'))
			# e2 is (64 x 64 x self.gf_dim*2)
			e3_real = self.g_bn_e3(conv2d(lrelu(e2_real), self.gf_dim*4, name='g_e3_conv_real'))
			# e3 is (32 x 32 x self.gf_dim*4)
			e4_real = self.g_bn_e4(conv2d(lrelu(e3_real), self.gf_dim*8, name='g_e4_conv_real'))
			# e4 is (16 x 16 x self.gf_dim*8)
			e5_real = self.g_bn_e5(conv2d(lrelu(e4_real), self.gf_dim*8, name='g_e5_conv_real'))
			# e5 is (8 x 8 x self.gf_dim*8)
			e6_real = self.g_bn_e6(conv2d(lrelu(e5_real), self.gf_dim*8, name='g_e6_conv_real'))
			# e6 is (4 x 4 x self.gf_dim*8)
			e7_real = self.g_bn_e7(conv2d(lrelu(e6_real), self.gf_dim*8, name='g_e7_conv_real'))
			# e7 is (2 x 2 x self.gf_dim*8)
			e8_real = self.g_bn_e8(conv2d(lrelu(e7_real), self.gf_dim*8, name='g_e8_conv'))
			# e8 is (1 x 1 x self.gf_dim*8)

			concat_feature = tf.concat([e8,e8_real],3)
			self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(concat_feature),
				[self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
			d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
			d1 = tf.concat([d1, e7], 3)
			# d1 is (2 x 2 x self.gf_dim*8*2)

			self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
				[self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
			d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
			d2 = tf.concat([d2, e6], 3)
			# d2 is (4 x 4 x self.gf_dim*8*2)

			self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
				[self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
			d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
			d3 = tf.concat([d3, e5], 3)
			# d3 is (8 x 8 x self.gf_dim*8*2)

			self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
				[self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
			d4 = self.g_bn_d4(self.d4)
			d4 = tf.concat([d4, e4], 3)
			# d4 is (16 x 16 x self.gf_dim*8*2)

			self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
				[self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
			d5 = self.g_bn_d5(self.d5)
			d5 = tf.concat([d5, e3], 3)
			# d5 is (32 x 32 x self.gf_dim*4*2)

			self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
				[self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
			d6 = self.g_bn_d6(self.d6)
			d6 = tf.concat([d6, e2], 3)
			# d6 is (64 x 64 x self.gf_dim*2*2)

			self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
				[self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
			d7 = self.g_bn_d7(self.d7)
			d7 = tf.concat([d7, e1], 3)
			# d7 is (128 x 128 x self.gf_dim*1*2)

			self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
				[self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
			# d8 is (256 x 256 x output_c_dim)

			return tf.nn.tanh(self.d8)

	def sampler(self, image, y=None):

		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()

			s = self.output_size
			s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

			# image is (256 x 256 x input_c_dim)
			e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
			# e1 is (128 x 128 x self.gf_dim)
			e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
			# e2 is (64 x 64 x self.gf_dim*2)
			e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
			# e3 is (32 x 32 x self.gf_dim*4)
			e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
			# e4 is (16 x 16 x self.gf_dim*8)
			e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
			# e5 is (8 x 8 x self.gf_dim*8)
			e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
			# e6 is (4 x 4 x self.gf_dim*8)
			e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
			# e7 is (2 x 2 x self.gf_dim*8)
			e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
			# e8 is (1 x 1 x self.gf_dim*8)

			# image is (256 x 256 x input_c_dim)
			e1_real = conv2d(real_image, self.gf_dim, name='g_e1_conv_real')
			# e1 is (128 x 128 x self.gf_dim)
			e2_real = self.g_bn_e2(conv2d(lrelu(e1_real), self.gf_dim*2, name='g_e2_conv_real'))
			# e2 is (64 x 64 x self.gf_dim*2)
			e3_real = self.g_bn_e3(conv2d(lrelu(e2_real), self.gf_dim*4, name='g_e3_conv_real'))
			# e3 is (32 x 32 x self.gf_dim*4)
			e4_real = self.g_bn_e4(conv2d(lrelu(e3_real), self.gf_dim*8, name='g_e4_conv_real'))
			# e4 is (16 x 16 x self.gf_dim*8)
			e5_real = self.g_bn_e5(conv2d(lrelu(e4_real), self.gf_dim*8, name='g_e5_conv_real'))
			# e5 is (8 x 8 x self.gf_dim*8)
			e6_real = self.g_bn_e6(conv2d(lrelu(e5_real), self.gf_dim*8, name='g_e6_conv_real'))
			# e6 is (4 x 4 x self.gf_dim*8)
			e7_real = self.g_bn_e7(conv2d(lrelu(e6_real), self.gf_dim*8, name='g_e7_conv_real'))
			# e7 is (2 x 2 x self.gf_dim*8)
			e8_real = self.g_bn_e8(conv2d(lrelu(e7_real), self.gf_dim*8, name='g_e8_conv'))
			# e8 is (1 x 1 x self.gf_dim*8)

			concat_feature = tf.concat([e8,e8_real],3)

			self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(concat_feature),
				[self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
			d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
			d1 = tf.concat([d1, e7], 3)
			# d1 is (2 x 2 x self.gf_dim*8*2)

			self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
				[self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
			d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
			d2 = tf.concat([d2, e6], 3)
			# d2 is (4 x 4 x self.gf_dim*8*2)

			self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
				[self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
			d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
			d3 = tf.concat([d3, e5], 3)
			# d3 is (8 x 8 x self.gf_dim*8*2)

			self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
				[self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
			d4 = self.g_bn_d4(self.d4)
			d4 = tf.concat([d4, e4], 3)
			# d4 is (16 x 16 x self.gf_dim*8*2)

			self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
				[self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
			d5 = self.g_bn_d5(self.d5)
			d5 = tf.concat([d5, e3], 3)
			# d5 is (32 x 32 x self.gf_dim*4*2)

			self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
				[self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
			d6 = self.g_bn_d6(self.d6)
			d6 = tf.concat([d6, e2], 3)
			# d6 is (64 x 64 x self.gf_dim*2*2)

			self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
				[self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
			d7 = self.g_bn_d7(self.d7)
			d7 = tf.concat([d7, e1], 3)
			# d7 is (128 x 128 x self.gf_dim*1*2)

			self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
				[self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
			# d8 is (256 x 256 x output_c_dim)

			return tf.nn.tanh(self.d8)

	def train(self):

		d_optim = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1) \
					.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1) \
					.minimize(self.g_loss, var_list=self.g_vars)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		self.g_sum = tf.summary.merge([self.d_filled_texture,
					self.fake_B_sum, self.d_loss_fake_sum_2, self.g_loss_sum])

		self.d_sum = tf.summary.merge([self.d_same_chair, self.d_different_chair, 
					self.d_loss_real_sum,self.d_loss_fake_sum_1, self.d_loss_sum])

		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

		for _ in range(args.epoch):

			_, summary_str = self.sess.run([d_optim, self.d_sum])
			self.writer.add_summary(summary_str, counter)
			_, summary_str = self.sess.run([g_optim, self.g_sum])
			self.writer.add_summary(summary_str, counter)
			_, summary_str = self.sess.run([g_optim, self.g_sum])
			self.writer.add_summary(summary_str, counter)

			counter += 1

			if np.mod(counter, args.sample_freq) == 1:
				self.sample_model(args.sample_dir, epoch, counter)

			if np.mod(counter, args.save_freq) == 2:
				self.save(args.checkpoint_dir, counter)                

	def save(self, checkpoint_dir, step):
		model_name = "pix2pix.model"
		model_dir = "%s_%s" % (self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)
		
	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s" % (self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def sample_model(self, sample_dir, epoch, idx):

		samples, d_loss, g_loss = self.sess.run(
			[self.fake_img_sample, self.d_loss, self.g_loss],
		)
		samples = (image + 1.)/2
		save_images(samples, 
					'./{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
		print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

