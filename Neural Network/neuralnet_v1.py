# This script was developed on Bazzite Linux 7.2.15-ogc1.1.fc45.x86_64
# Gemini 3.1 Pro assisted in debugging and code cleanup (see comments)
# Handouts from CSMC 191 Neural Networks, as well as the book 'Neural Networks From Scratch In Python" by Kinsley & Kukiela
# 	were used to assist in the creation of this script
#
# Author: Borromeo, Daniel William R.
# Date Created  : 10 November 2026
# Date Modified : 21 November 2026
# Version       : 1.1.0
import math, random, json

class NeuralNetwork:
	def __init__(self, input_dim, hidden_dim, output_dim, direct_connections=False, kappa=0.1, phi=0.5, theta=0.7, mu=0.9):
		self.ins  = input_dim
		self.hids = hidden_dim
		self.outs = output_dim

		self.direct = direct_connections

		self.kappa = kappa
		self.phi   = phi
		self.theta = theta
		self.mu    = mu

		self.a = [[0.2 * (random.random() - 0.5) for _ in range(self.hids)] for _ in range (self.ins + 1)] # hidden weights
		totalOutWeights = self.hids + self.ins + 1 if self.direct else self.hids + 1
		self.b = [[0.2 * (random.random() - 0.5) for _ in range(self.outs)] for _ in range(totalOutWeights)]
		# Gemini changed output weights to small random values so it doesn't saturate early

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

		self._reset_gradients()

	def _reset_gradients(self):
		self.dHid = [[0.0 for _ in range(self.hids)] for _ in range(self.ins + 1)]
		self.dOut = [[0.0 for _ in range(self.outs)] for _ in range(len(self.b))]
		self.nAccum = 0

	def _logistic(self, uv): # bounded sigmoid
		if uv < -88.0: return 0.0
		elif uv > 88.0: return 1.0
		else: return 1.0 / (1.0 + math.exp(-uv))

	def compute_forward(self, xSample):
		y = [0.0] * self.hids # hidden layer forward pass
		for j in range(self.hids):
			u = self.a[0][j] # Bias weight
			for i in range(self.ins):
				u += self.a[i+1][j] * xSample[i]
			y[j] = self._logistic(u)

		z = [0.0] * self.outs # output layer forward pass
		for k in range(self.outs):
			v = self.b[0][k] # Bias weight
			for j in range(self.hids):
				v += self.b[j+1][k] * y[j]

			if self.direct:
				for i in range(self.ins):
					v += self.b[self.hids+1+i][k] * xSample[i]
			
			z[k] = self._logistic(v)

		return y, z

	def accumulate_gradients(self, xSample, tSample, y, z):
		p = [0.0] * self.outs
		q = [0.0] * self.hids
		self.nAccum += 1

		for k in range(self.outs): # computes OL derivatives
			p[k] = (z[k] - tSample[k]) * z[k] * (1.0 - z[k])
			self.dOut[0][k] += p[k] # output bias gradient

			for j in range(self.hids):
				self.dOut[j+1][k] += p[k] * y[j]
				q[j] += p[k] * self.b[j+1][k]

			if self.direct:
				for i in range(self.ins):
					self.dOut[self.hids+1+i][k] += p[k] * xSample[i]

		for j in range(self.hids): # computes HL derivatives
			q[j] = q[j] * y[j] * (1.0 - y[j])
			self.dHid[0][j] += q[j] # hidden bias gradient

			for i in range(self.ins):
				self.dHid[i+ 1][j] += q[j] * xSample[i]

	def _update_weight_rule(self, w, d, f, e, c):
		if d * f > 0.0: e = min(e+self.kappa, 10.0)
		else: e *= self.phi

		f = (1.0 - self.theta) * d + self.theta * f
		c = -(1.0 - self.mu) * e * d + self.mu * c
		w += c

		return w, f, e, c

	def apply_epoch_changes(self):
		n = max(self.nAccum, 1)
		for i in range(self.ins + 1): # updates HL
			for j in range(self.hids):
				res = self._update_weight_rule(self.a[i][j], self.dHid[i][j]/n, self.fHid[i][j], self.eHid[i][j], self.cHid[i][j])
				self.a[i][j], self.fHid[i][j], self.eHid[i][j], self.cHid[i][j] = res

		totalOutWeights = len(self.b)
		for j in range(totalOutWeights): # updates OL
			for k in range(self.outs):
				res = self._update_weight_rule(self.b[j][k], self.dOut[j][k]/n, self.fOut[j][k], self.eOut[j][k], self.cOut[j][k])
				self.b[j][k], self.fOut[j][k], self.eOut[j][k], self.cOut[j][k] = res
			
		self._reset_gradients()

	def mse(self, X, T):
		s = 0.0
		for x, t in zip(X, T):
			_, z = self.compute_forward(x)
			s += sum((zk - tk) ** 2 for zk, tk in zip(z, t)) / self.outs
		return s / len(X)

	def fit(self, X, T, epochs=500):
		for _ in range(epochs):
			for x, t in zip(X, T):
				y, z = self.compute_forward(x)
				self.accumulate_gradients(x, t, y, z)
			self.apply_epoch_changes()

	def predict(self, x):
		return self.compute_forward(x)[1]

	def save(self, path, extra=None):
		with open(path, 'w') as fh:
			json.dump(
				{
					'ins': self.ins,
					'hids': self.hids,
					'outs': self.outs,
					'direct': self.direct,
			        'a': self.a,
					'b': self.b,
					'extra': extra
				}, fh
			)

	@classmethod
	def load(cls, path):
		with open(path) as fh: s = json.load(fh)
		net = cls(s['ins'], s['hids'], s['outs'], direct_connections=s['direct'])
		net.a, net.b = s['a'], s['b']
		return net, s.get('extra')