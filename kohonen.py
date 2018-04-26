import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from random import random, seed

seed()

class MapaIris():

	def __init__(self, taxa, dimensao, fi0, nVar):
		self._taxa = taxa
		self._dimensao = dimensao
		self._fi = fi0
		self._fi0 = fi0
		self._matriz = np.array([[-1 for i in range(self._dimensao)] for j in range(self._dimensao)])
		self.pesos = np.array([[[random() for i in range(nVar)] for j in range(self._dimensao)] for k in range(self._dimensao)])

		self._epoca = 1

	def _resetMatriz(self):
		self._matriz = np.array([[-1 for i in range(self._dimensao)] for j in range(self._dimensao)])

	def _atualizaTaxa(self):
		self._taxa = self._taxa * np.exp(-self._epoca/1000)

	def _atualizaFi(self):
		self._fi = self._fi * np.exp(-self._epoca/(1000/np.log(self._fi0)))

	def _vizinhanca(self, entrada, x, y):
		return np.exp(-np.linalg.norm(entrada - self.pesos[x][y])/(2*self._fi*self._fi))

	def _mudaEpoca(self):
		self._atualizaTaxa()
		self._atualizaFi()
		self._epoca += 1

	def _vencedor(self, entrada):
		i = 0;
		j = 0;
		_xvencedor = 0
		_yvencedor = 0
		menor = np.linalg.norm(entrada - self.pesos[0][0])
		achou = False
		while(not achou and i < self._dimensao):
			while(not achou and j < self._dimensao):
				if(self._matriz[i][j] == -1):
					achou = True
					menor = np.linalg.norm(entrada - self.pesos[i][j])
					_xvencedor = i
					_yvencedor = j
				j = j+1
			i = i+1

		for i in range(self._dimensao):
			for j in range(self._dimensao):
				#print(menor, ' > ',  np.linalg.norm(entrada - self.pesos[i][j]), ' ', (j + i*8), ' ', menor > np.linalg.norm(entrada - self.pesos[i][j]))
				if(menor > np.linalg.norm(entrada - self.pesos[i][j])):
					if(self._matriz[i][j] == -1):
						menor = np.linalg.norm(entrada - self.pesos[i][j])
						_xvencedor = i
						_yvencedor = j

		return([_xvencedor,_yvencedor])

	def treino(self, dTreino, saidas, periodo=1):
		for iteracao in range(periodo):
			self._resetMatriz()
			_posicao = 0
			for vTreino in dTreino:
				#print(vTreino, " saida: ", saidas[_posicao])
				[x, y] = self._vencedor(vTreino)
				viz = self._vizinhanca(vTreino, x, y)
				# procura o menor dos pesos e atualiza pesos
				self.pesos[x][y] = self.pesos[x][y] + self._taxa* viz *(vTreino - self.pesos[x][y])
				
				self._matriz[x,y] = saidas[_posicao]

				_posicao = _posicao + 1

			self._mudaEpoca()

	def teste(self, entrada):
		menor = np.linalg.norm(entrada - self.pesos[0][0])
		_xvencedor = 0
		_yvencedor = 0
		for i in range(self._dimensao):
			for j in range(self._dimensao):
				#print(menor, ' > ',  np.linalg.norm(entrada - self.pesos[i][j]), ' ', (j + i*8), ' ', menor > np.linalg.norm(entrada - self.pesos[i][j]))
				if(menor > np.linalg.norm(entrada - self.pesos[i][j])):
					menor = np.linalg.norm(entrada - self.pesos[i][j])
					_xvencedor = i
					_yvencedor = j

		return(self._matriz[_xvencedor][_yvencedor])

	def getMatriz(self):
		return self._matriz


dados = datasets.load_iris()

dado_setosa = dados['data'][0]
dado_versicolor = dados['data'][50]
dado_virginica = dados['data'][100]

mapa = MapaIris(.1,13,8,4)


kf = KFold(n_splits=10, shuffle=True, random_state=None)
acuracia = []
for iteracao in range(10):
	y_verdadeiro = []
	y_previsto = []
	print('\n')
	print("Iteracao: ", iteracao)
	for indices_treino, indices_teste in kf.split(dados['data']):
		mapa.treino(dados['data'][indices_treino], dados['target'][indices_treino])
		for i in indices_teste:
			y_previsto.append(mapa.teste(dados['data'][i]))
			y_verdadeiro.append(dados['target'][i])
	print('Matriz de confusao:')
	print(confusion_matrix(y_verdadeiro, y_previsto, [0,1,2]))
	acuracia.append(accuracy_score(y_verdadeiro, y_previsto))
	print('Acuracia: ', accuracy_score(y_verdadeiro, y_previsto))


acuraciaMedia = 0
for a in acuracia:
	acuraciaMedia = acuraciaMedia + a

acuraciaMedia = acuraciaMedia/len(acuracia)
print("\n\nAcuracia Media: ", acuraciaMedia)
