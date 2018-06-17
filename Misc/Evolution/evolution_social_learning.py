# okay, so the aim here is to figure uot waht's going on with this
# so our model should be simple - fniite population, individuals in the population - 
# testing first uniparental transmission - i.e. each parent has a generation and has cultural traits
# first essentially have a trait or have no trait
# and then figure out how to get it to work... who knows?
# how shold I represent this in terms of data structures?
# so essentially just do it with gaussians to be honest!
# so each agent develops from parent. has an x percent chance of developing it
# so essentially has a probability of developing social learning
# could simply do it as a list of lists, that seems reasonable! and wuold just be binary
# i.e. learn or don't learn, and this would be very simple to implement in a basic way
# and could lead to some kind of interesing results
# and the maintenance of culture

# so each trait has a fitness is the general number of offspring, taken from a normal distribtion
# so does the lack of the trait
# then each offspring has a probability of the trait given by p which itself is selected from a learning readiness
# which is the key thing - and the learning readiness which is sampled from a normal distribution centered
# around the current normal distribution and those more likely to learn will generally have the trait

import numpy as np
import matplotlib.pyplot as plt

#general parameters
no_trait_mean = 1
no_trait_variance=1

trait_mean = 1.5
trait_variance = 1

initial_learnability = 0.5
learnability_variance = 0.5

initial_trait_probability = 0.4

initial_population_number = 100000

def uniform():
	return np.random.uniform(low=0, high=1)

def normal(mu, var):
	return np.random.normal(mu, var)

def positive_normal(mu, var):
	val = np.random.normal(mu, var)
	if val <0:
		return 0
	return val

def generate_initial_population(N, initial_trait_probability):
	init_pop = []
	for i in xrange(N): #this is horrendously inefficient!
		rand = uniform()
		if rand <= initial_trait_probability:
			init_pop.append([1, initial_learnability])
		else:
			init_pop.append([0, initial_learnability])
	return init_pop

def evolve_generation(pop_list):
	new_gen = []
	for trait, orig_learnability in pop_list:
		num_offspring = 0
		if trait == 1:
			num_offspring =int(positive_normal(trait_mean, trait_variance))
		if trait == 0:
			num_offspring = int(positive_normal(no_trait_mean, no_trait_variance))
		# now generate the actual offspring!
		for i in xrange(num_offspring):
			new_learnability = positive_normal(orig_learnability, learnability_variance)
			new_gen.append([trait, new_learnability])
	print "new generation population: " + str(len(new_gen))
	return new_gen

#this is actually an incredibly simple solution which is just crazy who easy it is to make an 'agent' based model of how it works!
def run_evolution(N, initial_size):
	pops = []
	init_pop = generate_initial_population(initial_size, initial_trait_probability)
	pops.append(init_pop)
	#now run the main experiment
	for i in xrange(N):
		print "generation: " + str(i) + " evolved"
		#run the experiment
		pops.append(evolve_generation(pops[i])) # I think this will work
	return pops

# now for some basic analysis functions!
def get_trait_prevalence(pops):
	prevalences = []
	variances  = []
	for i in xrange(len(pops)):
		prevalences.append(np.mean(pops[i][0]))
		variances.append(np.var(pops[i][0]))
	return prevalences, variances

def get_learnabilities_prevalences(pops):
	prevalences = []
	variances  = []
	for i in xrange(len(pops)):
		prevalences.append(np.mean(pops[i][1]))
		variances.append(np.var(pops[i][1]))
	return prevalences, variances

def plot_prevalences(prevalences):
	plt.plot(prevalences)
	plt.show()


#now run it to see what happens

if __name__ == '__main__':
	pops = run_evolution(100, 100000)
	tprevs, tvars = get_trait_prevalence(pops)
	lprevs, lvars = get_learnabilities_prevalences(pops)
	plot_prevalences(tprevs)
	plot_prevalences(lprevs)

#it's amazing that this seems to work so straightforwardly, and its pretty crazy
## and total uniparental transform, so who knows if it actually learns
# that's quite cool,  but who knows
# assuming social learning can itself evolve successfully
# that's pretty amazing, so who knows?

