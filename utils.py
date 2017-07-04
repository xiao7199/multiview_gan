import numpy as np
from PIL import Image
import caffe,time
import glob, os
import scipy.misc



def caffe_chair_segmentation(net, image_path, chair_index = 9, crop_size = 473,pad_val_h = 512,pad_val_w = 512,nb_class = 21):

	file_name_counter_recorder = {}
	mean_val = np.array([123.68,116.779,103.939])
	file_name_list = [f for f in os.listdir(image_path) if f.endswith('.JPG')]
	for file_name in file_name_list:
		file_prefix = file_name[:len(file_name)-6]
		file_counter = file_name[file_name.index('_')+1:file_name.index('.')]
		if file_name in file_name_counter_recorder:
			file_name_counter_recorder[file_name] += 1
		else:
			file_name_counter_recorder[file_name] = 1

	reverse_palette = reverse_pascal_palette()
	print('[o] start processing the chair segmentation')
	img_counter = 0
	for file_name in file_name_list:

		img = Image.open(os.path.join(image_path,file_name))
		img = img.convert('RGB')
		img = np.array(img, dtype = np.float32)
		stride = 1.0 * crop_size

		img-= mean_val
		img = img[:,:,::-1]
		img_h,img_w,_ = img.shape

		if img_h <= crop_size and img_w <= crop_size:
			pad_image = np.zeros((crop_size,crop_size,3))
			pad_image[:img_h,:img_w,:] = img
			net.blobs['data'].data[...] = pad_image.transpose(2,0,1)
			net.forward()
			pred_img = net.blobs['conv6_interp'].data[0].transpose(1,2,0)
			pred_img = np.exp(pred_img)/np.sum(np.exp(pred_img), axis = 2, keepdims = True)
			segment_result = pred_img.argmax(2)	
		else:
			pad_image = np.zeros((pad_val_h,pad_val_w,3))

			pad_image[:img_h,:img_w,:] = img
			input_img = pad_image
			h_time = np.int32(np.ceil((1.0 * pad_val_h - crop_size) / stride))+1
			w_time = np.int32(np.ceil((1.0 * pad_val_w - crop_size) / stride))+1
			counter = np.zeros((pad_val_h,pad_val_w,nb_class))
			data_record = np.zeros((pad_val_h,pad_val_w,nb_class))
			for i in range(h_time):
				for j in range(w_time):
					s_x = int(i * stride)
					s_y = int(j * stride)
					e_x = min(s_x + crop_size, pad_val_h)
					e_y = min(s_y + crop_size, pad_val_w)
					s_x = e_x - crop_size
					s_y = e_y - crop_size
					counter[s_x:e_x,s_y:e_y,:] += 1
					img_sub = input_img[s_x:e_x,s_y:e_y,:]
					img_sub = img_sub.transpose(2,0,1)
					net.blobs['data'].data[...] = img_sub
					net.forward()
					pred_img = net.blobs['conv6_interp'].data[0].transpose(1,2,0)
					data_record[s_x:e_x,s_y:e_y,:]\
									 += np.exp(pred_img)/np.sum(np.exp(pred_img), axis = 2, keepdims = True)

			data_record /= counter
			segment_result = data_record[:img_h,:img_w,:]
			segment_result = segment_result.argmax(axis = 2)
		img_counter += 1
		if np.mod(img_counter,50)==0:
			print(img_counter)
		segmented_chair = np.zeros((img_h,img_w,3))
		for i in range(img_h):
			for j in range(img_w):
				segmented_chair[i,j,:] = reverse_palette.get(segment_result[i,j])
		scipy.misc.imsave(image_path+'/'+file_name[:len(file_name)-3]+'png',segmented_chair.astype(np.uint8))

	print('[*] caffe segmentation is complete')

def get_masked_image(image, mask, pascal_palette ,chair_index = [9,18], return_mask = False):

	[h,w,_] = image.shape
	binary_mask_record = np.ones((h,w)) == 1
	for current_chair_index in chair_index:
		current_color = pascal_palette.get(current_chair_index)
		current_binary_mask = (mask[:,:,0] == current_color[0]) \
				& (mask[:,:,1] == current_color[1]) & (mask[:,:,2] == current_color[2])
		binary_mask_record &= current_binary_mask
	image[binary_mask_record,:] = np.zeros((1,3))
	if return_mask is False:
		return image
	else:
		mask[binary_mask_record,:] = np.zeros((1,3))
		return image,mask,binary_mask_record

def compute_mean_val(image_path):
	print('[O] Start computing the mean value')
	file_name_list = [f for f in os.listdir(image_path) if f.endswith('.JPG')]
	mean_val = np.zeros((1,3))
	img_num = len(file_name_list)
	for image_name in file_name_list:
		img = np.array(Image.open(os.path.join(image_path,image_name)).convert('RGB')).astype(np.float32)
		mean_val += np.mean(img, axis = (0,1))
	mean_val /= img_num
	print('[*] Finish computing the mean value (RGB format): {}'.format(mean_val))
	return mean_val

def get_file_name_counter_hashmap(image_path):
	file_name_counter_recorder = {}
	file_name_list = [f for f in os.listdir(image_path) if f.endswith('.JPG')]
	for file_name in file_name_list:
		file_prefix = file_name[:file_name.index('_')]
		file_counter = file_name[file_name.index('_')+1:file_name.index('.')]
		if file_prefix in file_name_counter_recorder:
			file_name_counter_recorder[file_prefix] += 1
		else:
			file_name_counter_recorder[file_prefix] = 1

	file_prefix_keys_array = file_name_counter_recorder.keys()
	return file_prefix_keys_array

def image_reader(image_path, image_size, batch_size, mean_val):
	#read image dataset and store the file name
	palette = reverse_pascal_palette() 
	file_name_counter_recorder = {}
	
	if mean_val is None:
		mean_val = compute_mean_val(image_path)

	file_name_list = [f for f in os.listdir(image_path) if f.endswith('.JPG')]
	for file_name in file_name_list:
		file_prefix = file_name[:file_name.index('_')]
		file_counter = file_name[file_name.index('_')+1:file_name.index('.')]
		if file_prefix in file_name_counter_recorder:
			file_name_counter_recorder[file_prefix] += 1
		else:
			file_name_counter_recorder[file_prefix] = 1

	file_prefix_keys_array = file_name_counter_recorder.keys()
	total_prefix_length = len(file_name_counter_recorder)

	batch_counter = 0 

	while 1:

		file_prefix_index_array = np.arange(total_prefix_length)
		np.random.shuffle(file_prefix_index_array)

		for current_prefix_index in file_prefix_index_array:

			if batch_counter == 0:
				image_array = np.zeros((3,batch_size,image_size,image_size,3))
				segmentation_array = np.zeros((batch_size,image_size,image_size,4))

			next_prefix_index = (current_prefix_index + np.random.randint(0,total_prefix_length-1)) % total_prefix_length
			
			current_prefix = file_prefix_keys_array[current_prefix_index]
			next_prefix = file_prefix_keys_array[next_prefix_index]
			random_2view_index = np.random.randint(0,file_name_counter_recorder[current_prefix],size = 2)
			random_1view_chair2_index = np.random.randint(0,file_name_counter_recorder[next_prefix],size = 1)
			
			chair1_view1_image_name = current_prefix+'_{}.JPG'.format(random_2view_index[0])
			chair1_view2_image_name = current_prefix+'_{}.JPG'.format(random_2view_index[1])
			chair2_image_name = next_prefix+'_{}.JPG'.format(random_1view_chair2_index[0])

			chair1_view1_image = Image.open(os.path.join(image_path,chair1_view1_image_name)).convert('RGB')
			chair1_view2_image = Image.open(os.path.join(image_path,chair1_view2_image_name)).convert('RGB')
			chair2_image = Image.open(os.path.join(image_path,chair2_image_name)).convert('RGB')

			chair1_view1_image = np.array(scipy.misc.imresize(chair1_view1_image, [image_size,image_size]),np.float32)
			chair1_view2_image = np.array(scipy.misc.imresize(chair1_view2_image, [image_size,image_size]),np.float32)
			chair2_image = np.array(scipy.misc.imresize(chair2_image, [image_size,image_size]),np.float32)

			chair1_view1_mask_image_name = current_prefix+'_{}.png'.format(random_2view_index[0])
			chair1_view2_mask_image_name = current_prefix+'_{}.png'.format(random_2view_index[1])
			chair2_mask_image_name = next_prefix+'_{}.png'.format(random_1view_chair2_index[0])

			chair1_view1_mask = Image.open(os.path.join(image_path,chair1_view1_mask_image_name))
			chair1_view2_mask = Image.open(os.path.join(image_path,chair1_view2_mask_image_name))
			chair2_mask = Image.open(os.path.join(image_path,chair2_mask_image_name))

			chair1_view1_mask = np.array(scipy.misc.imresize(chair1_view1_mask, [image_size,image_size]),np.float32)
			chair1_view2_mask = np.array(scipy.misc.imresize(chair1_view2_mask, [image_size,image_size]),np.float32)
			chair2_mask = np.array(scipy.misc.imresize(chair2_mask, [image_size,image_size]),np.float32)

			chair1_view1_mask_image = get_masked_image(chair1_view1_image,chair1_view1_mask,palette)
			chair1_view2_mask_image,chair1_view2_mask,chair1_view2_binary_mask \
					= get_masked_image(chair1_view2_image,chair1_view2_mask,palette,return_mask = True)
			chair2_mask_image = get_masked_image(chair2_image,chair2_mask,palette)


			image_array[0,batch_counter,:,:,:] =  chair1_view1_mask_image - mean_val
			image_array[1,batch_counter,:,:,:] =  chair1_view2_mask_image - mean_val
			image_array[2,batch_counter,:,:,:] =  chair2_mask_image - mean_val
#			scipy.misc.imsave('img.png', chair1_view2_mask)
			segmentation_array[batch_counter,:,:,:3] = chair1_view2_mask - mean_val
			segmentation_array[batch_counter,:,:,3:4] = np.expand_dims(chair1_view2_binary_mask , axis = 2)
			batch_counter += 1

			if batch_counter == batch_size:
				batch_counter = 0
				yield image_array,segmentation_array


def save_images(images, image_path):
	return scipy.misc.imsave(image_path, images[0,:,:,:])

def reverse_pascal_palette():
	palette = {0:(  0,   0,   0)  ,
				1:(128,   0,   0)  ,
				2:(  0, 128,   0)  ,
				3:(128, 128,   0)  ,
				4:(  0,   0, 128)  ,
				5:(128,   0, 128)  ,
				6:(  0, 128, 128)  ,
				7:(128, 128, 128)  ,
				8:( 64,   0,   0)  ,
				9:(192,   0,   0)  ,
				10:( 64, 128,   0)  ,
				11:(192, 128,   0)  ,
				12:( 64,   0, 128)  ,
				13:(192,   0, 128)  ,
				14:( 64, 128, 128)  ,
				15:(192, 128, 128)  ,
				16:(  0,  64,   0)  ,
				17:(128,  64,   0)  ,
				18:(  0, 192,   0)  ,
				19:(128, 192,   0)  ,
				20:(  0,  64, 128)  }
	return palette

def pascal_palette():
	palette = {(  0,   0,   0) : 0 ,
			 (128,   0,   0) : 1 ,
			 (  0, 128,   0) : 2 ,
			 (128, 128,   0) : 3 ,
			 (  0,   0, 128) : 4 ,
			 (128,   0, 128) : 5 ,
			 (  0, 128, 128) : 6 ,
			 (128, 128, 128) : 7 ,
			 ( 64,   0,   0) : 8 ,
			 (192,   0,   0) : 9 ,
			 ( 64, 128,   0) : 10,
			 (192, 128,   0) : 11,
			 ( 64,   0, 128) : 12,
			 (192,   0, 128) : 13,
			 ( 64, 128, 128) : 14,
			 (192, 128, 128) : 15,
			 (  0,  64,   0) : 16,
			 (128,  64,   0) : 17,
			 (  0, 192,   0) : 18,
			 (128, 192,   0) : 19,
			 (  0,  64, 128) : 20 }
	return palette
