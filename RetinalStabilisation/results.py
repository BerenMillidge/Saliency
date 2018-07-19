from __future__ import division
import numpy as np 
import scipy
import keras
from keras.models import load_model
from utils import *
from plotting import *


def calculate_average_error(augments_name, copies_name, save_name=None):
	augment_test = np.load(augments_name + '_test.npy')
	augment_preds = np.load(augments_name+'_preds.npy')

	print augment_test.shape
	print augment_preds.shape

	copy_test = np.load(copies_name + '_test.npy')
	copy_preds = np.load(copies_name + '_preds.npy')

	print copy_test.shape
	print copy_preds.shape

	print np.max(augment_test)

	augment_errmaps = get_error_maps(augment_test, augment_preds)
	copy_errmaps = get_error_maps(copy_test, copy_preds)
	augment_error = get_mean_error(augment_errmaps)
	copy_error =get_mean_error(copy_errmaps)

	print "mnist augments error: " , augment_error
	print "mnist copies error: " , copy_error

	if save_name is not None:
		save_array([augment_error, copy_error], save_name)
		
	return augment_error, copy_error

	#the only thign I can think is that there's somethingdifferent about the test set
	# but I have no idea hwat

	#this looks identical. how come sthere is so much error... are the preds just insane?
	# I'm so confused
	# I guess in results  I should just loop through to see if anything obviously disastrous is happengin!

	#so in this case the results are saved, btu what happens if I actually go through and try to figure some
	# additoinal things out in this case? - i.e. try to predict them from scratch - deos it break again
	# must I have trained the invariances from the model?

def calculate_average_error_from_models(augments_name, copies_name, augments_model, copy_model,save_name=None):
	if isinstance(augments_name, str):
		augment_test = np.load(augments_name + '_test.npy')
	#augment_preds = np.load(augments_name+'_preds.npy')
	# else assume it is augments
	else:
		augments_test = augments_name

	print augment_test.shape
	#print augment_preds.shape

	if isinstance(copies_name, str):
		copy_test = np.load(copies_name + '_test.npy')
	#copy_preds = np.load(copies_name + '_preds.npy')
	else:
		copy_test = copies_name

	auggment_model = load_model(augments_model)
	copy_model = load_model(copy_model)
	aug_preds = auggment_model.predict(augment_test)
	copy_preds = copy_model.predict(augment_test)

	print copy_test.shape
	print copy_preds.shape

	print np.max(augment_test)

	augment_errmaps = get_error_maps(augment_test, aug_preds)
	copy_errmaps = get_error_maps(augment_test, copy_preds)
	augment_error = get_mean_error(augment_errmaps)
	copy_error =get_mean_error(copy_errmaps)

	print "mnist augments error: " , augment_error
	print "mnist copies error: " , copy_error

	if save_name is not None:
		save_array([augment_error, copy_error], save_name)
		
	return augment_error, copy_error

	# let's test this and see what happens, it is very possible that the test functions are failing
	# and that the models aren't saved properly or something which is why it gives different results
	# for theo the predictions than here, so this is to eliminate another possible issue with these results!

	#okay, so what I need to do is ultimately test if there is a difference here, and if there is do something aboutit
	# and I don't know what. I need to finesse the data sufficiently to get this thing into a publishable state
	# hopefully else all the work I put into this would have been wasted, which is really really really bad
	# I also need to look at reading all about the predictive processing and trying to figure it out
	# going through fristons stuff and then being able to eventually try to recreate Rao and Ballard's  work, hopefully
	# sufficient to at least 

	# the trouble is that this tests everythin on eaach ones data while I need to test the copy model on the 
	# more variable model to see what's up there! but I just don't know!


	# okay, so my thing is I've got to figure out what precisely I need


#results: 
#mnist augments error:  0.0458422217494
#mnist copies error:  1.46164913036

#validation errors written down for the copies:
#in form epoch: train_loss, val_loss
#1: 0.0222, 0.0100
#2: 0.0124, 0.0083
#3: 0.0109, 0.0075
#4: 0.0100, 0.0070
# 5: 0.0094, 0.0067
# 6: 0.0089, 0.0064
#7@ 00086, 0.0062
#8: 0.0083, 0.0060
#9: 0.0081, 0.0058
#10: 0.0079, 0.0057

def save_history_losses(his_fname, save_fname):
	his = load(his_fname)
	history = his['history']
	his = None # free history
	loss = history['loss']
	val_loss = history['val_loss']
	print type(loss)
	print type(val_loss)
	np.save(save_fname+'_training_loss',loss)
	np.save(save_fname+'_validation_loss',val_loss)
	print "saved"
	return loss, val_loss


def test_results():
	aug_his = load('mnist_augments_history')
	val_data = aug_his['validation_data']
	#print len(val_data)
	#print val_data[0].shape
	print aug_his.keys()
	history = aug_his['history']
	print "history"
	print type(history)
	print len(history)
	print history.keys()
	loss = history['loss']
	val_loss = history['val_loss']
	print type(loss)
	print len(loss)
	print type(val_loss)
	print len(val_loss)
	params = aug_his['params']
	print "params"
	print type(params)
	print len(params)
	print params.keys()
	epoch = aug_his['epoch']
	print "epoch"
	print type(epoch)
	print len(epoch)
	print "validation data"
	print type(val_data)
	print len(val_data)


def test_fixations():
	copy_results = load('results/from_scratch_fixation_copy')
	aug_results =load('results/from_scratch_fixation_augments')
	print type(copy_results)
	print len(copy_results)
	plot_fixation_errmaps(aug_results, copy_results)


def get_mean_total_error(errmaps):
	N = len(errmaps)
	total = 0
	for errmap in errmaps:
		total += np.sum(errmap)
	return total/N


def test_generative_invariance(aug_model, copy_model, results_save=None, plot=False):
	# this tests the ivnariance of the model - i.e. how good it is at predicting
	# the whole image it is given for the different invairances tested#
	# I should also do a classificatory invariance as well, althoguh it would mean
	# training an entirely separate invariant classification, so that could work well too!

	#load the models
	# I'm also not sure why the random walk drift model seems to make everything so difficult or why the
	# other one has SO much more error! I'm just not seeing the issue here and it's so confusing!?

	# so the issue almost certainly comes from the normalization. I guess the next thing to figure out is
	#what is actually a real problem and what is not then?
	# I can rerun everything with the normaliation, but the ultimate issue is that the copy model is spatially invariant
	# in a fact it probably should be because of the CNN architecture,... dagnabbit! that's really stupid, I should have thought of that
	# but without that, what should I do? what would be a reasonable thing to do here?
	# the question is, then, why do the other models show a serious improvement for the error of the augmented
	# model over the copy model if the invariant oen doesn't... perhaps it is just all in the normalisation?
	aug_model = load_model(aug_model)
	copy_model = load_model(copy_model)

	#load the invariance files
	mnist_0px = np.load('data/mnist_invariance_0pixels_translate.npy')[:,:,:] # make the problem managable!
	mnist_2px = np.load('data/mnist_invariance_2pixels_translate.npy')[:,:,:]
	mnist_4px = np.load('data/mnist_invariance_4pixels_translate.npy')[:,:,:]
	mnist_6px = np.load('data/mnist_invariance_6pixels_translate.npy')[:,:,:]
	mnist_8px = np.load('data/mnist_invariance_8pixels_translate.npy')[:,:,:]

	#print mnist_0px.shape
	#print mnist_2px.shape

	pixels = [0,2,4,6,8]
	# so this supposedly can't explain it... who knows?
	##for pixel in pixels:
	#	print pixel

	# I think the error issues might be caused by a lack of normalisation perhaps
	# the other problem is that it seems that the copy model is actually completely spatially invariant
	# despite being not trained for the invariance at all and develops stronger responeses... but who knows?



	invariances = [mnist_0px,mnist_2px,mnist_4px,mnist_6px,mnist_8px]

	aug_errors = []
	copy_errors = []

	#test on each and get errors
	#for invariance in invariances:
	for i in xrange(len(invariances)):
		invariance = invariances[i] # see if this is causing me any issues!
		invariance = invariance.astype('float64')/255.
		#reshape
		sh = invariance.shape
		invariance = np.reshape(invariance, (sh[0], sh[1], sh[2],1))

		aug_preds = aug_model.predict(invariance)
		copy_preds = copy_model.predict(invariance)
		s = aug_preds.shape
#
		#aug_preds = np.reshape(aug_preds,(s[0], s[1],s[2]))
		#copy_preds = np.reshape(copy_preds,(s[0], s[1],s[2]))

		#print aug_preds.shape


		#plot them
		if plot:
			for i in xrange(20):
				print np.max(invariance[i])
				print np.max(aug_preds[i])
				fig = plt.figure()
				ax1 = fig.add_subplot(131)
				plt.imshow(np.reshape(invariance[i], (s[1], s[2])))
				ax2 = fig.add_subplot(132)
				print "aug pred"
				plt.imshow(np.reshape(aug_preds[i],(s[1],s[2])))
				ax3 = fig.add_subplot(133)
				print "copy pred"
				plt.imshow(np.reshape(copy_preds[i],(s[1],s[2])))
				plt.show()
		aug_errmaps = get_error_maps(invariance, aug_preds)
		copy_errmaps = get_error_maps(invariance, copy_preds)
		#aug_mean_error = get_mean_total_error(aug_errmaps)
		#copy_mean_error = get_mean_total_error(copy_errmaps)

		aug_mean_error = get_mean_error(aug_errmaps)
		copy_mean_error = get_mean_error(copy_errmaps)
		# and hopefully this will be the cause of the discrepancy, because I can't see anything else really!?

		#the other thing I can think is that the data is just not the same!?
		# maybe my invariancegenerating code is just wrong so it always comes out silly
		# what if I sub in the model test set into that to see wha happens
		# that's something I should try because I vaguely know what results I want!?
		# so I just don't know!


		aug_errors.append(aug_mean_error)
		copy_errors.append(copy_mean_error)

		#maybe the problem is smoething really weird in how the means are calculated. I@m gong to try
		# to calculate the mean in both ways to see what's up there to see what happens
		# but who knwos really?

		# okay, so what steps are necessary here, and what do I say... the trouble is, its had for me to say anything except that
		# its currently notworkign and I need the results ASAP and I simply don't have them at the moment... dagnabbit
		# and I@m really far behind and haven't really achieved anything this month, and its really frustrating and Idon't know what todo
		# and it just rankles and upsets me, especially when we have nights like tonight which are totally wasted!

	aug_errors = np.array(aug_errors)
	copy_errors = np.array(copy_errors)

	if results_save:
		np.save(results_save+'_aug', aug_errors)
		np.save(results_save+'_copy', copy_errors)
		np.save(results_save+'_pixels', pixels)
	return aug_errors, copy_errors,pixels

def test_discriminative_invariance(aug_model, copy_model, results_save=None, info=False):
	aug_model = load_model(aug_model)
	copy_model = load_model(copy_model)

	mnist_0px_data = np.load('data/discriminative_0pixels_translate_data.npy')
	mnist_2px_data = np.load('data/discriminative_2pixels_translate_data.npy')
	mnist_4px_data = np.load('data/discriminative_4pixels_translate_data.npy')
	mnist_6px_data = np.load('data/discriminative_6pixels_translate_data.npy')
	mnist_8px_data = np.load('data/discriminative_8pixels_translate_data.npy')

	#I'm going to convert these to one-hot - do that here
	mnist_0px_labels = np.load('data/discriminative_0pixels_translate_labels.npy')
	mnist_2px_labels = np.load('data/discriminative_2pixels_translate_labels.npy')
	mnist_4px_labels = np.load('data/discriminative_4pixels_translate_labels.npy')
	mnist_6px_labels = np.load('data/discriminative_6pixels_translate_labels.npy')
	mnist_8px_labels = np.load('data/discriminative_8pixels_translate_labels.npy')

	pixels = [0,2,4,6,8]

	invariance_data = [mnist_0px_data,mnist_2px_data,mnist_4px_data,mnist_6px_data,mnist_8px_data]
	invariance_labels = [mnist_0px_labels,mnist_2px_labels,mnist_4px_labels,mnist_6px_labels,mnist_8px_labels]

	aug_accuracies = []
	copy_accuracies = []

	for i in range(len(invariance_data)):
		data = invariance_data[i]
		#reshape
		sh = data.shape
		data = np.reshape(data, (sh[0],sh[1],sh[2],1))

		#reshape labels
		labels = one_hot(invariance_labels[i])
		#predict
		aug_pred_labels = aug_model.predict(data)
		copy_pred_labels = copy_model.predict(data)
		print "predictions"
		print aug_pred_labels.shape
		#just get teh accuracies and save it
		aug_acc = classification_accuracy(labels, aug_pred_labels)
		copy_acc = classification_accuracy(labels, copy_pred_labels)

		print "pixels: " +str(pixels[i])
		print "aug acc: " , aug_acc
		print "copy acc: " , copy_acc

		aug_accuracies.append(aug_acc)
		copy_accuracies.append(copy_acc)

	aug_accuracies = np.array(aug_accuracies)
	copy_accuracies = np.array(copy_accuracies)
	# so this is quite strange, the accuracies here are just terrible
	# I wonder if the network can actuallylearn anything useful here
	# at least the pattern works. I think I'll need to learn it with like
	#50 epochs instead see if it helps hopefully!

	if results_save is not None:
		np.save(results_save+'_aug_accuracies', aug_accuracies)
		np.save(results_save+'_copy_accuracies', copy_accuracies)
		np.save(results_save+'_pixels', pixels)
	return aug_accuracies, copy_accuracies, pixels


def split_cross_validate_invariance_accuracies(aug_model, copy_model, num_splits=10, results_save=None, plot=False):
	if type(aug_model) is str:
		aug_model = load_model(aug_model)
	if type(copy_model) is str:
		copy_model = load_model(copy_model)

	mnist_0px_data = np.load('data/discriminative_0pixels_translate_data.npy')
	mnist_2px_data = np.load('data/discriminative_2pixels_translate_data.npy')
	mnist_4px_data = np.load('data/discriminative_4pixels_translate_data.npy')
	mnist_6px_data = np.load('data/discriminative_6pixels_translate_data.npy')
	mnist_8px_data = np.load('data/discriminative_8pixels_translate_data.npy')

	#I'm going to convert these to one-hot - do that here
	mnist_0px_labels = np.load('data/discriminative_0pixels_translate_labels.npy')
	mnist_2px_labels = np.load('data/discriminative_2pixels_translate_labels.npy')
	mnist_4px_labels = np.load('data/discriminative_4pixels_translate_labels.npy')
	mnist_6px_labels = np.load('data/discriminative_6pixels_translate_labels.npy')
	mnist_8px_labels = np.load('data/discriminative_8pixels_translate_labels.npy')



	pixels = [0,2,4,6,8]

	invariance_data = [mnist_0px_data,mnist_2px_data,mnist_4px_data,mnist_6px_data,mnist_8px_data]
	invariance_labels = [mnist_0px_labels,mnist_2px_labels,mnist_4px_labels,mnist_6px_labels,mnist_8px_labels]

	all_aug_accs = []
	all_copy_accs =[]
	assert len(invariance_data) == len(invariance_labels),'Serious problem. invariance labels and data different shape'
	for i in xrange(len(invariance_data)):
		data = invariance_data[i]
		labels = invariance_labels[i]
		#these are parralel arrays so this should work
		aug_accs = []
		copy_accs = []
		split_length = len(data)//num_splits
		for j in xrange(num_splits):
			d = data[i*split_length:(j+1)*split_length]
			l = labels[i*split_length:(j+1)*split_length]
			sh = d.shape
			d = np.reshape(d, (sh[0],sh[1],sh[2],1))

			#reshape labels
			l = one_hot(l)
			#predict
			aug_pred_labels = aug_model.predict(d)
			copy_pred_labels = copy_model.predict(d)
			# this  is just totally rubbish since it is discriminative
			# but I don't want discriminative I want generative. I can't relaly be bothered
			# to figure this out now!
			print "predictions"
			print aug_pred_labels.shape
			#just get teh accuracies and save it
			print l.shape
			print aug_pred_labels.shape
			aug_acc = classification_accuracy(l, aug_pred_labels)
			copy_acc = classification_accuracy(l, copy_pred_labels)

			print "split number: " + str(j) + "pixels: " +str(pixels[i])
			print "aug acc: " , aug_acc
			print "copy acc: " , copy_acc

			aug_accs.append(aug_acc)
			copy_accs.append(copy_acc)
		aug_accs = np.array(aug_accs)
		copy_accs = np.array(copy_accs)
		#now analysis and printing
		print "pixels: " + str(pixels[i]) + " analysis:"
		print "aug mean: " + str(np.mean(aug_accs))
		print "copy mean: ", np.mean(copy_accs)
		print "aug median: ", median(aug_accs)
		print "copy median: ", median(copy_accs)
		print "aug variance: ", np.var(aug_accs)
		print "copy variance: ", np.var(copy_accs)
		if plot:
			fig = plt.figure()
			plt.bar(aug_accs, label='Augmentation accuracies')
			plt.bar(copy_accs, label='Copy accuracies')
			plt.legend()
			plt.show()
		all_aug_accs.append(aug_accs)
		all_copy_accs.append(copy_accs)

	#now save
	if save_name is not None:
		save_array(all_aug_accs, save_name+'_all_aug_accs')
		save_array(all_copy_accs, save_name+'_all_copy_accs')

	return all_aug_accs, all_copy_accs


	#this cross-validation approach would be nice if I wanted error bars, which I should get, but first thigns first
	# just the very simple invariances and fix them
	# the only thing I can think isthat the 

	#so, the copy and the generative network are tested on different train sets... WHY ISTHIS!?
	# this is actually really weird and sadly it means that the main thing is probably wrong
	# but I just don't understand why... argh
	#let's compare then, shall I

def compare_test_sets(t1, t2,N):
	t1 = np.load(t1)
	t2 = np.load(t2)
	print t1.shape
	print t2.shape
	sh  = t1.shape
	for i in xrange(N):
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		plt.imshow(np.reshape(t1[i], (sh[1], sh[2])))
		ax2 = fig.add_subplot(122)
		plt.imshow(np.reshape(t2[i], (sh[1], sh[2])))
		plt.show()


def crossvalidate_average_errors(augments_name, copies_name, num_split, split_size, augment_model, copy_model, save_name):
	#just assume everything is fine and load it!
	aug_train = np.load(augments_name + '_train.npy')
	aug_test = np.load(augments_name + '_test.npy')
	copy_train = np.load(augments_name + '_train.npy')
	copy_test = np.load(augments_name _+ '_test.npy')

	aug_data = np.concatenate((aug_train, aug_test))
	copy_data = np.concatenate((copy_train, copy_train))
	print aug_data.shape
	print copy_data.shape

	assert len(aug_data) == len(copy_data),'data of aug and copy data must be the same'
	l = len(aug_data)

	augerrs = []
	copyerrs = []
	for i in xrange(num_split):
		#generate the random number
		rand = l
		while rand < l - split_size:
			rand = l * np.random.uniform(low=0, high=1)

		# okay, got correct rand
		aug_dat = aug_data[rand:rand+split_size]
		copy_dat = copy_data[rand:rand+split_size]
		aug_err, copy_err = calculate_average_error_from_models(aug_dat, copy_dat, augment_model, copy_model)
		augerrs.append(aug_err)
		copyerrs.append(copy_err)

	augerrs = np.array(augerrs)
	copyerrs =np.array(copyerrs)

	if save_name is not None:
		np.save(save_name + '_augerrs', augerrs)
		np.save(save_name + '_copyerrs', copyerrs)

	return augerrs, copyerrs

# so this is one crossvalidation function... what about others!?


#and some quick tests
if __name__ == '__main__':
	
	#print "In main!"
	#test_results()
	#calculate_average_error('mnist_augments', 'mnist_copies', save_name="errors_1")
	#plot_errmaps('mnist_augments', 'mnist_copies')
	#save_history_losses('mnist_augments_history', 'augments')
	#save_history_losses('mnist_copies_history','copies')

	#test_fixations()
	#test_generative_invariance('model_mnist_augments', 'model_mnist_copy','results/generative_invariance')
	#test_discriminative_invariance('discriminative_aug_model_2','discriminative_copy_model_2', 'results/discriminative_invariance_2')
	# it sort of shows waht I want to show, but not that well, dagnabbit!

	#begin with the drift!
	#calculate_average_error('results/drift_aug', 'results/drift_copy', save_name='drift_errors_1')

	#let's find and cross-validate the accuracies first:
	#test_generative_invariance('models/microsaccade_or_copy_model', 'models/copy_model',results_save='new_results/accuracies/microsaccade_or_copy_crossval_accuracies_3')
	#test_generative_invariance('models/drift_and_microsaccades_model', 'models/copy_model',  results_save='new_results/accuracies/microsaccade_and_drift_crossval_accuracies_3')
	#test_generative_invariance('models/random_walk_drift_model', 'models/copy_model',results_save='new_results/accuracies/drift_random_walk_crossval_accuracies_3')



	#now for the standard errors!
	#calculate_average_error('new_results/microsaccade_or_copy_aug','new_results/copy_aug', save_name='new_results/errors/microsaccade_or_copy_errors')
	#calculate_average_error('new_results/drift_and_microsaccades_aug','new_results/copy_aug', save_name='new_results/errors/drift_and_microsaccade_errors')
	#calculate_average_error('new_results/random_walk_drift_aug','new_results/copy_aug', save_name='new_results/errors/random_walk_drift_errors')

	# I am pretty scared... if this doesn'twork them I'm fairly doomed as I'll ahve no paper to actually read/write
	# which certainly isn't ideal. having three papers by the end of the first year at minimum isn't best!

	#compare_test_sets('new_results/copy_aug_test.npy', 'new_results/drift_and_microsaccades_aug_test.npy',50)
	calculate_average_error_from_models('new_results/drift_and_microsaccades_aug','new_results/copy_aug','models/drift_and_microsaccades_model','models/copy_model', save_name='new_results/model_averages/drift_and_microsaccades_proper')
	calculate_average_error_from_models('new_results/microsaccade_or_copy_aug','new_results/copy_aug','models/microsaccade_or_copy_model','models/copy_model', save_name='new_results/model_averages/microsaccade_or_copy_proper')
	calculate_average_error_from_models('new_results/random_walk_drift_aug','new_results/copy_aug','models/random_walk_drift_model','models/copy_model', save_name='new_results/model_averages/random_walk_drift_proper')
	# I now changed this ot calculate the errors of the copy model wrt the agument datasets, which seems reasonable!

	# okay, while that is running let's dothe survey I've been procrastinating out of for ages!

	# okay, so this is really weird... why is the copy network better!? at dealing with invariances in the input than the network trained at that/???
	# that's the reuslt I'm getting and it's weird! dagnabbit! I need to figuer this out!
	# why is thi sstill not working, so effectively it just proves that its better at its own thing, but why/

	# as I don' think I'm going to realistically end up doing any library work tonight
	# at least not immediately, I should just work on getting a cross validator together
	# to try ou tvarious things!