# this is where I just put really random things which are easy to find here. so we can inspect data and so forth. this isgoig to be a very freeform file. for scripting, as that's what python is meant to do.

import numpy as np
import scipy
from file_reader import *
from utils import *
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

"""
img = read_image(50)
#print img.shape
#show_colour_splits(img)

# okay, let's try out the fft stuff
img_freq = np.fft.fft2(img)
#calculate amplitude spectum
img_amp = np.fft.fftshift(np.abs(img_freq))
#for display take the logarithm so the scale isn't so small
img_amp_display = np.log(img_amp + 0.0001)
#rescale to -1:+1 for display
img_amp_display = (((img_amp_display - np.min(img_amp_display))*2)/np.ptp(img_amp_display)) -1

print type(img_amp_display)
print img_amp_display.shape

print img
print "  "
print "  "
print img_amp_display

img_amp_display = img_amp_display *255
img_amp_display =  img_amp_display.astype('uint8')
plt.imshow(img_amp_display)
plt.show()

"""
"""
# let's test our high pass filtering
#get_amplitude_spectrum(img, show=True)
#get_magnitude_spectrum(img, show=True)
# reshape
print img.shape
# okay, let's get this as grayscale
img = img[:,:,0]
img = reshape_into_image(img)
print img.shape

#hpf = high_pass_filter(img, show=True)
#print hpf

# okay, other band tests here:
plt.imshow(img)
plt.show()
print "High Pass!"
hpf = highpass_filter(img, show=True)
print "  "
print "Low Pass"
lpf = lowpass_filter(img, show=True)
print "  "
print "Bandpass"
bpf = bandpass_filter(img, show=True)

print "both"
compare_two_images(lpf, hpf, 'Low Pass Filter', 'High Pass Filter')
"""
"""
# okay, we're going to test the filesystem stuff here
rootdir = './testSet/Stimuli'
#print_dirs_files(rootdir)
save_images_per_directory(rootdir, save=False)

"""
"""
# okay, tests on image splitting

(xtrain, ytrain), (xtest, ytest)  = cifar10.load_data()
redtrain = xtrain[:,:,:,0]
print redtrain.shape
redtrain = np.reshape(redtrain,(len(redtrain), 32,32))
print redtrain.shape
half1, half2 = split_image_dataset_into_halves(redtrain)
for i in xrange(10):
	compare_two_images(half1[i], half2[i])

"""

# tests get files saved
"""
rootdir = './BenchmarkIMAGES/SM'
cropsize = (200,200)
save_images_per_directory(rootdir, cropsize)"""

# now let's check this actuallt works, we can load them and get images, and so forth, that seems rather important to me
def load_file_test(fname):
	imgs = load(fname)
	print type(imgs)
	print imgs.shape
	plt.imshow(imgs[3])
	plt.show()

#load_file_test('BenchmarkIMAGES_images')
#load_file_test('BenchmarkIMAGES_output')

#rootdir = 'testSet/Stimuli/Action/'
#make_dir = 'testSet_Arrays_Action'
#save_images_per_directory(rootdir, save=True, crop_size=(100,100), make_dir_name=make_dir)

#load_file_test(make_dir + '/testSet_images')
#load_file_test(make_dir + '/Action_output')

#rootdir = 'BenchmarkIMAGES/'
#make_dir = 'BenchmarkDATA'
#save_images_per_directory(rootdir, save=True, crop_size=(200,200), make_dir_name=make_dir)


#rootdir = 'testSet_Arrays'
#make_dir = 'combined'
#combine_arrays_into_one(rootdir, make_dir_name=make_dir)

#load_file_test('testSet_Arrayscombined/_combined')

#imgs = load('testSet_Arrayscombined/images_combined')
#for i in xrange(20):	
#	plt.imshow(imgs[200+i])
#	plt.show()

def get_files_in_directory(dirname):
	filelist = []
	for fname in sorted(os.listdir(dirname)):
		filelist.append(fname)
		print fname
	return filelist

def get_dirs_from_rootdir(rootdir, mode='RGB', crop_size = None, save=True, save_dir=None):

	if save:
		assert save_dir is not None and type(save_dir) == str, 'save directory must exist and be a string'

	for dirs, subdirs, files in os.walk(rootdir):
		print dirs
		filelist = get_files_in_directory(dirs)
		arr = []
		for f in filelist:
			#first we check it's a file
			if '.' in f:
				#tehn we get rid of the jpg
				fname = f.split('.')[0]
				#next we check if it's output or not
				# it's output
				if crop_size is not None:
					img = imresize(imread(dirs+ '/'+f, mode=mode), crop_size)
				if crop_size is None:
					img = imread(dirs+'/'+fname, mode=mode)
				arr.append(img)

		arr = np.array(arr)
		splits = dirs.split('/')
		name='default'
		if len(splits) == 3:
			# i.e. normal file
			name= splits[-1]
		if len(splits) == 4:
			name= splits[-2] + '_' + splits[-1]
		if save:
			save_array(arr, save_dir + '/' + name)
			print "SAVING AS: " + str(save_dir + '/' + name)
				
				


# okay, the sorted is the key, without that it just breaks terribly. So we need to fix this in our functoins before it works, which should hopefully help, so let's work at that! now at least we nkow the problem. hopefully we can get some reasonable results tomorrow to show to richard, for thursday

#dirname = 'testSet/Stimuli/Action'
#get_files_in_directory(dirname)

#rootdir = 'testSet/Stimuli'
#get_dirs_from_rootdir(rootdir, crop_size = (100, 100), save_dir = 'testSet/Data/test')

#load_file_test('testSet/Data/test/Action')
#load_file_test('testSet/Data/test/Action_Output')

def compare_image_and_salience(dirname, N=20, start=0):
	for i in xrange(N):
		imgs = load(dirname)
		imgs = imgs[:,:,:,0]
		shape = imgs.shape
		imgs = np.reshape(imgs, (shape[0], shape[1], shape[2]))
		print "IMGS:"
		print imgs.shape
		outputs = load(dirname+'_Output')
		shape = outputs.shape
		outputs = outputs[:,:,:,0]
		outputs = np.reshape(outputs, (shape[0], shape[1], shape[2]))
		print "OUTPUTS:"
		print outputs.shape
		compare_images((imgs[start+i], outputs[start+i]), ('image', 'salience map'))

def compare_image_and_salience_from_known_files(fname1, fname2, N=20, start=0):
	imgs = load(fname1)
	imgs = imgs[:,:,:,0]
	shape = imgs.shape
	imgs = np.reshape(imgs, (shape[0], shape[1], shape[2]))
	print "IMGS:"
	print imgs.shape
	outputs = load(fname2)
	shape = outputs.shape
	outputs = outputs[:,:,:,0]
	outputs = np.reshape(outputs, (shape[0], shape[1], shape[2]))
	print "OUTPUTS:"
	print outputs.shape
	for i in xrange(N):
		compare_images((imgs[start+i], outputs[start+i]), ('image', 'salience map'))

#compare_image_and_salience('testSet/Data/test/Action')

compare_image_and_salience_from_known_files('testimages_combined', 'testsaliences_combined')

def combine_images_into_big_array(dirname, makedir = '', save=True, verbose=True):	

	if makedir != '':
		if not os.path.exists(rootdir + makedir):
			try:
				os.makedirs(rootdir + makedir)
			except OSError as e:
				if e.errno!= errno.EEXIST:
					print "error found: " + str(e)
					raise
				else:
					print "directory probably already exists despite check"
					raise
		

	filelist =  sorted(os.listdir(dirname))
	imgs = []
	outputs = []
	for f in filelist:
		arr = load(dirname + '/' + f)
		if '_' in f: #i.e. it's an output
			outputs.append(arr)
			print "OUTPUT: " + f
			print arr.shape
		if '_' not in f: # so it's an image
			imgs.append(arr)
			print "IMAGE: " + f
			print arr.shape
	#we now stack them

	imgs = np.concatenate(imgs)
	outputs = np.concatenate(outputs)
	if verbose:
		print "images shape: " + str(imgs.shape)
		print "outputs shape: " + str(outputs.shape)

	if save: 
		save_array(imgs, dirname + makedir + 'images_combined')
		save_array(outputs, dirname + makedir + 'saliences_combined')

	return imgs, outputs

#dirname = 'testSet/Data/test'
#combine_images_into_big_array(dirname)



#plt.imshow(img_amp_display)
#plt.show()
"""
error_maps = load('error_map_test')
error_map = error_maps[1]
print error_map.shape
err = np.reshape(error_map, [28,28])
print err.shape
plt.imshow(err)
plt.show()"""

"""
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
print xtrain.shape

red = xtrain[:,:,:,0]
print red.shape
redimg = red[5]
print redimg.shape


plt.imshow(xtrain[5])
plt.show()

plt.imshow(redimg)
plt.show()
"""






"""
a = [[1,5,3],[4,5,7],[7,5,2]]
a = np.array(a)
print a
print a.shape
print np.argmax(a)
print np.argmax(a, axis=0)
print np.argmax(a, axis=1)"""




