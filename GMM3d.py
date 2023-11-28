import numpy as np
import math

#Gaussian Mixture Model
class GMM:
	'''The GMM: Gaussian Mixture Model algorithm'''
	'''Each point in the image belongs to a GMM, and because each pixel owns
		three channels: RGB, so each component owns three means, 9 covs and a weight.'''
	
	def __init__(self, k = 5):
		'''k is the number of components of GMM'''
		self.k = k
		self.weights = np.asarray([0. for i in range(k)], dtype = 'float32') # Weight of each component
		self.means = np.asarray([0. for i in range(k)], dtype = 'float32') # Means of each component
		self.vars = np.asarray([0. for i in range(k)], dtype = 'float32') # vars of each component

		self.pixel_counts = np.asarray([0 for i in range(k)]) # Count of pixels in each components
		self.pixel_total_count = 0 # The total number of pixels in the GMM
		
		# The following parameter is assistant parameters for counting pixels and calc. params.
		self._sums = np.asarray([0. for i in range(k)])

	def _prob_pixel_component(self, pixel, ci):
		'''Calculate the probability of each pixel belonging to the ci_th component of GMM'''
		'''Using the formula of multivariate normal distribution'''
		return 1/np.sqrt(self.vars[ci]) * np.exp(-0.5*math.pow(pixel-self.means[ci], 2)/self.vars[ci]) # gaussian distribution formula

	def prob_pixel_GMM(self, pixel):	
		'''Calculate the probability of each pixel belonging to this GMM, which is the sum of 
			the prob. of the pixel belonging to each component * the weight of the component'''
		'''Also the first term of Gibbs Energy(negative;)'''
		return sum([self._prob_pixel_component(pixel, ci) * self.weights[ci] for ci in range(self.k)])

	def most_likely_pixel_component(self, pixel):
		'''Calculate the most likely component that the pixel belongs to'''
		prob = np.asarray([self._prob_pixel_component(pixel, ci) for ci in range(self.k)])
		return prob.argmax(0)
	
	def learning(self, components):
		for ci in range(self.k):
			print('length of conponent[',ci,']:', len(components[ci]))
			self.means[ci] = components[ci].mean()
			self.vars[ci] = components[ci].var()
			if self.vars[ci] < 0.1:
				self.vars[ci] += 1
			self.pixel_counts[ci] = components[ci].size

		self.pixel_total_count = self.pixel_counts.sum()
		for ci in range(self.k):
			self.weights[ci] = self.pixel_counts[ci]/self.pixel_total_count