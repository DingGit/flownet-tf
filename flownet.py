import tensorflow as tf
import flownetUtils as fu

def FlowNet(combine_image, ground_truth, isTraining):
	# conv1
	with tf.name_scope('conv1'):
		W_conv1 = fu.weight_variable([7, 7, 6, 64]) 
		b_conv1 = fu.bias_variable([64])
		h_conv1 = fu.conv2d(combine_image, W_conv1, [1, 2, 2, 1]) + b_conv1
		conv1 = fu.lrelu(h_conv1)

	# conv2
	with tf.name_scope('conv2'):
		W_conv2 = fu.weight_variable([5, 5, 64, 128]) 
		b_conv2 = fu.bias_variable([128])
		h_conv2 = fu.conv2d(conv1, W_conv2, [1, 2, 2, 1]) + b_conv2
		conv2 = fu.lrelu(h_conv2)
                                   
	# conv3
	with tf.name_scope('conv3'):
		W_conv3 = fu.weight_variable([5, 5, 128, 256]) 
		b_conv3 = fu.bias_variable([256])
		h_conv3 = fu.conv2d(conv2, W_conv3, [1, 2, 2, 1]) + b_conv3
		conv3 = fu.lrelu(h_conv3) 

	# conv3_1
	with tf.name_scope('conv3_1'):
		W_conv3_1 = fu.weight_variable([3, 3, 256, 256]) 
		b_conv3_1 = fu.bias_variable([256])
		h_conv3_1 = fu.conv2d(conv3, W_conv3_1, [1, 1, 1, 1]) + b_conv3_1
		conv3_1 = fu.lrelu(h_conv3_1) 

	# conv4
	with tf.name_scope('conv4'):
		W_conv4 = fu.weight_variable([3, 3, 256, 512]) 
		b_conv4 = fu.bias_variable([512])
		h_conv4 = fu.conv2d(conv3_1, W_conv4, [1, 2, 2, 1]) + b_conv4
		conv4 = fu.lrelu(h_conv4) 

	# conv4_1
	with tf.name_scope('conv4_1'):
		W_conv4_1 = fu.weight_variable([3, 3, 512, 512]) 
		b_conv4_1 = fu.bias_variable([512])
		h_conv4_1 = fu.conv2d(conv4, W_conv4_1, [1, 1, 1, 1]) + b_conv4_1
		conv4_1 = fu.lrelu(h_conv4_1) 

	# conv5
	with tf.name_scope('conv5'):
		W_conv5 = fu.weight_variable([3, 3, 512, 512]) 
		b_conv5 = fu.bias_variable([512])
		h_conv5 = fu.conv2d(conv4_1, W_conv5, [1, 2, 2, 1]) + b_conv5
		conv5 = fu.lrelu(h_conv5) 

	# conv5_1
	with tf.name_scope('conv5_1'):
		W_conv5_1 = fu.weight_variable([3, 3, 512, 512]) 
		b_conv5_1 = fu.bias_variable([512])
		h_conv5_1 = fu.conv2d(conv5, W_conv5_1, [1, 1, 1, 1]) + b_conv5_1 
		conv5_1 = fu.lrelu(h_conv5_1)

	# conv6
	with tf.name_scope('conv6'):
		W_conv6 = fu.weight_variable([3, 3, 512, 1024]) 
		b_conv6 = fu.bias_variable([1024])
		h_conv6 = fu.conv2d(conv5_1, W_conv6, [1, 2, 2, 1]) + b_conv6
		conv6 = fu.lrelu(h_conv6)

	# conv6_1
	with tf.name_scope('conv6_1'):
		W_conv6_1 = fu.weight_variable([3, 3, 1024, 1024]) 
		b_conv6_1 = fu.bias_variable([1024])
		h_conv6_1 = fu.conv2d(conv6, W_conv6_1, [1, 1, 1, 1]) + b_conv6_1
		conv6_1 = fu.lrelu(h_conv6_1) 


	# pr6 + loss6
	with tf.name_scope('pr6_loss6'):
		W_pr6 = fu.weight_variable([3, 3, 1024, 2]) 
		b_pr6 = fu.bias_variable([2])
		pr6 = fu.conv2d(conv6_1, W_pr6, [1, 1, 1, 1]) + b_pr6
		gt6 = tf.image.resize_nearest_neighbor(ground_truth, np.int32([ground_truth.get_shape().as_list()[1]/64, ground_truth.get_shape().as_list()[2]/64]))
		loss6 = fu.loss(pr6, gt6)

	# upconv5
	with tf.name_scope('upconv5'):
		W_upconv5 = fu.weight_variable([4, 4, 512, 1024]) 
		b_upconv5 = fu.bias_variable([512])
		h_upconv5 = fu.upconv2d_2x2(conv6_1, W_upconv5, [FLAGS.batch_size, np.int32(HEIGHT / 32), np.int32(WIDTH / 32), 512]) + b_upconv5 
		upconv5 = fu.lrelu(h_upconv5)
	
	# upsample_flow6to5
	with tf.name_scope('upsample_flow6to5'):
		W_upflo65 = fu.weight_variable([4, 4, 2, 2])
		b_upflo65 = fu.bias_variable([2])
		upflo65 = fu.upconv2d_2x2(pr6, W_upflo65, [FLAGS.batch_size, np.int32(HEIGHT / 32), np.int32(WIDTH / 32), 2]) + b_upflo65
		
	# concat5
	with tf.name_scope('concat5'):
		concat5 = tf.concat(axis=3, values=[conv5_1, upconv5, upflo65])
		
	# pr5 + loss5
	with tf.name_scope('pr5_loss5'):
		W_pr5 = fu.weight_variable([3, 3, 512+512+2, 2]) 
		b_pr5 = fu.bias_variable([2])
		pr5 = fu.conv2d(concat5, W_pr5, [1, 1, 1, 1]) + b_pr5
		gt5 = tf.image.resize_nearest_neighbor(ground_truth, np.int32([ground_truth.get_shape().as_list()[1]/32, ground_truth.get_shape().as_list()[2]/32]))
		loss5 = fu.loss(pr5, gt5)

	# upconv4
	with tf.name_scope('upconv4'):
		W_upconv4 = fu.weight_variable([4, 4, 256, 512+512+2])
		b_upconv4 = fu.bias_variable([256])
		h_upconv4 = fu.upconv2d_2x2(concat5, W_upconv4, [FLAGS.batch_size, np.int32(HEIGHT / 16), np.int32(WIDTH / 16), 256]) + b_upconv4
		upconv4 = fu.lrelu(h_upconv4)

	# upsample_flow5to4
	with tf.name_scope('upsample_flow5to4'):
		W_upflo54 = fu.weight_variable([4, 4, 2, 2])
		b_upflo54 = fu.bias_variable([2])
		upflo54 = fu.upconv2d_2x2(pr5, W_upflo54, [FLAGS.batch_size, np.int32(HEIGHT / 16), np.int32(WIDTH / 16), 2]) + b_upflo54
		
	# concat4
	with tf.name_scope('concat4'):
		concat4 = tf.concat(axis=3, values=[conv4_1, upconv4, upflo54])

	# pr4 + loss4
	with tf.name_scope('pr4_loss4'):
		W_pr4 = fu.weight_variable([3, 3, 512+256+2, 2]) 
		b_pr4 = fu.bias_variable([2])
		pr4 = fu.conv2d(concat4, W_pr4, [1, 1, 1, 1]) + b_pr4
		gt4 = tf.image.resize_nearest_neighbor(ground_truth, np.int32([ground_truth.get_shape().as_list()[1]/16, ground_truth.get_shape().as_list()[2]/16]))
		loss4 = fu.loss(pr4, gt4)

	# upconv3
	with tf.name_scope('upconv3'):
		W_upconv3 = fu.weight_variable([4, 4, 128, 512+256+2]) 
		b_upconv3 = fu.bias_variable([128])
		h_upconv3 = fu.upconv2d_2x2(concat4, W_upconv3, [FLAGS.batch_size, np.int32(HEIGHT / 8), np.int32(WIDTH / 8), 128]) + b_upconv3
		upconv3 = fu.lrelu(h_upconv3)
	
	# upsample_flow4to3
	with tf.name_scope('upsample_flow4to3'):
		W_upflo43 = fu.weight_variable([4, 4, 2, 2])
		b_upflo43 = fu.bias_variable([2])
		upflo43 = fu.upconv2d_2x2(pr4, W_upflo43, [FLAGS.batch_size, np.int32(HEIGHT / 8), np.int32(WIDTH / 8), 2]) + b_upflo43

	# concat3
	with tf.name_scope('concat3'):
		concat3 = tf.concat(axis=3, values=[conv3_1, upconv3, upflo43])

	# pr3 + loss3
	with tf.name_scope('pr3_loss3'):
		W_pr3 = fu.weight_variable([3, 3, 256+128+2, 2]) 
		b_pr3 = fu.bias_variable([2])
		pr3 = fu.conv2d(concat3, W_pr3, [1, 1, 1, 1]) + b_pr3 
		gt3 = tf.image.resize_nearest_neighbor(ground_truth, np.int32([ground_truth.get_shape().as_list()[1]/8, ground_truth.get_shape().as_list()[2]/8]))
		loss3 = fu.loss(pr3, gt3)

	# upconv2
	with tf.name_scope('upconv2'):
		W_upconv2 = fu.weight_variable([4, 4, 64, 256+128+2]) 
		b_upconv2 = fu.bias_variable([64])
		h_upconv2 = fu.upconv2d_2x2(concat3, W_upconv2, [FLAGS.batch_size, np.int32(HEIGHT / 4), np.int32(WIDTH / 4), 64]) + b_upconv2
		upconv2 = fu.lrelu(h_upconv2)

	# upsample_flow3to2
	with tf.name_scope('upsample_flow3to2'):
		W_upflo32 = fu.weight_variable([4, 4, 2, 2])
		b_upflo32 = fu.bias_variable([2])
		upflo32 = fu.upconv2d_2x2(pr3, W_upflo32, [FLAGS.batch_size, np.int32(HEIGHT / 4), np.int32(WIDTH / 4), 2]) + b_upflo32
		
	# concat2
	with tf.name_scope('concat2'):
		concat2 = tf.concat(axis=3, values=[conv2, upconv2, upflo32])
	
	# pr2 + loss2
	with tf.name_scope('pr2_loss2'):
		W_pr2 = fu.weight_variable([3, 3, 128+64+2, 2]) 
		b_pr2 = fu.bias_variable([2])
		pr2 = fu.conv2d(concat2, W_pr2, [1, 1, 1, 1]) + b_pr2 
		gt2 = tf.image.resize_nearest_neighbor(ground_truth, np.int32([ground_truth.get_shape().as_list()[1]/4, ground_truth.get_shape().as_list()[2]/4]))
		loss2 = fu.loss(pr2, gt2)
	
	# overall loss
	with tf.name_scope('loss'):
		total_loss = (0.005*loss2 + 0.01*loss3 + 0.02*loss4 + 0.08*loss5 + 0.32*loss6)
	
	return pr2, total_loss
