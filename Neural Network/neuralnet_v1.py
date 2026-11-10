# This script was developed on Bazzite Linux 7.2.15-ogc1.1.fc45.x86_64
# No AI tools were used in the development of this script
# Handouts from CSMC 191 Neural Networks, as well as the book 'Neural Networks From Scratch In Python" by Kinsley & Kukiela
#  were used to assist in the creation of this script
#
# Author: Borromeo, Daniel William R.
# Date Created  : 10 November 2026
# Date Modified : 10 November 2026
# Version       : 1.0.0

import numpy as np
import math, random

class NeuralNetwork:
	def __init__(self, input_dim, hidden_dim, output_dim, direct_connections=false, kappa=0.1, phi=0.5, theta=0.7, mu=0.9):
		self.ins  = input_dim
		self.hids = hidden_dim
		self.outs = output_dim

		self.direct = direct_connections

		self.kappa = kappa
		self.phi   = phi
		self.theta = theta
		self.mu    = mu

		self.a = [[0.2 * (random.random() - 0.5) for _ in range(self.hids)] for _ in range (self.ins + 1)] # hidden weights
		self.b = [] # out weights

		totalOutWeights = self.hids + self.ins + 1 if self.direct else self.hids + 1
		for j in range(totalOutWeights):
			row = []
			for k in range(self.outs):
				if J <= self.hids : row.append(1.0 if j%2==0 else -1.0)
				else : row.append(0.2 * (random.random() - 0.5))
			self.b.append(row)

		# The below are adaptive paramters and memory structures
		self.eHid = [[self.kappa for _ in range(self.hids)] for _ in range(self.ins + 1)]
		self.eOut = [[self.kappa for _ in range(self.outs)] for _ in range(totalOutWeights)]
		self.fHid = [[0.0 for _ in range(self.hids)] for _ in range(self.ins + 1)]
		self.fOut = [[0.0 for _ in range(self.outs)] for _ in range(totalOutWeights)]
		self.cHid = [[0.0 for _ in range(self.hids)] for _ in range(self.ins + 1)]
		self.cOut = [[0.0 for _ in range(self.outs)] for _ in range(totalOutWeights)]

		# Cumulative epoch gradients
		self.dHid = [[0.0 for _ in range(self.hids)] for _ in range(self.ins + 1)]
		self.dOut = [[0.0 for _ in range(self.outs)] for _ in range(totalOutWeights)]
