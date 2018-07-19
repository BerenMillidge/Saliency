# right, yeah... ultimate aim here is to get the animation side of this working properly... who knwos if that will
# actually ever work... totally not sure, and who knows!?

# so just animate or give graphs of the microsaccade condition

import numpy as np
#from utils import * 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation


# get drift data
driftdata = np.load('data/random_walk_drift_train.npy')
# load first ten
ten = driftdata[0:10]
print ten.shape 
#for i in xrange(10):
##	plt.imshow(ten[i])
#	plt.show()

def animate(dat, save_name=None):
	fig = plt.figure()
	plt.xticks([])
	plt.yticks([])
	im = plt.imshow(dat[0], animated=True, aspect='auto')
	plt.title('Copy 0')

	def updateFig(i):
		im.set_array(dat[i-1])
		title = plt.title('Copy ' + str(i))
		return im,

	length = len(dat)
	plt.subplots_adjust(wspace=0, hspace=0)

	anim = animation.FuncAnimation(fig, updateFig, interval=22, blit=True, save_count=length)
	anim.save(save_name,writer="ffmpeg", fps=5, extra_args=['-vcodec', 'libx264'])

def animate_weight_matrix(weightlist,save_name):

	if not isinstance(save_name, str):
		raise TypeError('Save name must be a string')
	if save_name.split('.')[-1] != 'mp4':
		save_name = save_name + '.mp4' # add the file encoding type on!
	fig = plt.figure()
	plt.xticks([])
	plt.yticks([])
	im = plt.imshow(weightlist[0], animated=True,cmap='gray')
	#plt.show(im)
	#print im.shape

	def updateFig(i):
		im.set_array(weightlist[i-1])
		title = plt.title('Epoch: ' + str(i))
		#plt.show(im)
		#print im.shape 
		# something has gone terribly wrong with the shape of the im here!
		print im
		return im,

	print len(weightlist)

	plt.subplots_adjust(wspace=0, hspace=0)
	anim = animation.FuncAnimation(fig, updateFig, interval=30, blit=True, save_count=10)
	anim.save(save_name,writer="ffmpeg", fps=3, extra_args=['-vcodec', 'libx264'])


# the issue seems to be something to do with the array being over before it can figure this out?

animate_weight_matrix(ten, 'animation/drift_example_1.mp4')

# okay, I'll probably have to work on cross-validatoin of some sort to get statistics for this paper
# I don' really have any idea how I would go about doing that... dagnabbit... so it goes
# nevertheless, could be interesting... who even knows!?
# okay, let's actually sort out the cross validation, if I can

