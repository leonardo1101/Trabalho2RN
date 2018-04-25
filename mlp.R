######################################################################################
## mlp-trabalho1.R																	##
## Primeiro trabalho para a disciplina de Tópicos de informática 8 (Redes Neurais)	##
## Departamento de Computação da Universidade Federal de São Carlos (DC-UFSCar)		##
## Autor: João Gabriel Melo Barbirato, RA: 726546									##
## Professor: Ricardo Cerri 														##
######################################################################################
#library(modelr)
# Funcao de ativacao
funcao.ativacao <- function(v){
	return(1 / (1+exp(-v)))
}

# Derivada da funcao de ativacao
der.funcao.ativacao <- function(y){
	return(y*(1 - y))
}

# >>> DIFERENTE <<<
# desclassificadorIsis
# Objetivo: transformar a string que classifica o dado em um vetor de 3 posições (uma para cada neurônio de saída)
# Entrada esperada: string da classe
# Saída: lista de 3 posições
desclassificadorIsis <- function(classe){
	if(classe == 'setosa'){ 
		return(c(1,0,0))
	}
	if(classe == 'versicolor'){
		return(c(0,1,0))
	}
	if(classe == 'virginica'){
		return(c(0,0,1))
	}
}# fim desclassificadorIsis

# Arquitetura (da rede: matrizes com pesos iniciais)
arquitetura <- function(num.entrada,  num.escondida, num.saida, funcao.ativacao,der.funcao.ativacao,desclassificador=NULL){

	arq <- list()
	arq$num.entrada <- num.entrada
	arq$num.escondida <- num.escondida
	arq$num.saida <- num.saida
	arq$funcao.ativacao <- funcao.ativacao
	arq$der.funcao.ativacao <- der.funcao.ativacao

	#>>> DIFERENTE <<<
	# funcao da arquitetura para quando a classe do dado está como um tipo
	# de difícil utilidade matemática (por exemplo, string)
	arq$desclassificador <- desclassificador

	# Pesos conectando a camada de entrada com a escondida
	num.pesos.entrada.escondida <- (num.entrada+1)*num.escondida
	arq$escondida <- matrix(runif(min=-0.5,max=0.5,num.pesos.entrada.escondida),nrow=num.escondida, num.entrada+1)
	# Pesos conectando a camada escondida com a saida
	num.pesos.escondida.saida <- (num.escondida+1)*num.saida
	arq$saida <- matrix(runif(min=-0.5, max=0.5, num.pesos.escondida.saida), nrow=num.saida,ncol=num.escondida+1)
	return(arq)
}

# Propagacao
mlp.propagacao <- function(arq,exemplo){
	# Entrada -> Escondida
	v.entrada.escondida <- arq$escondida %*% as.numeric(c(exemplo,1)) # pondo o bias
	y.entrada.escondida <- arq$funcao.ativacao(v.entrada.escondida)

	# Escondida -> saida
	v.escondida.saida <- arq$saida %*% c(y.entrada.escondida,1)

	# >>> DIFERENTE <<<
	# é necessário aplicar a função de ativação em todos os campos do vetor de saida
	y.escondida.saida <- unlist(lapply(v.escondida.saida, arq$funcao.ativacao))

	# Resultados
	resultados <- list()
	resultados$v.entrada.escondida <- v.entrada.escondida
	resultados$y.entrada.escondida <- y.entrada.escondida
	resultados$v.escondida.saida <- v.escondida.saida
	resultados$y.escondida.saida <- y.escondida.saida

	return(resultados)
}


# Retro-propagacao (limiar= criterio de parada)
mlp.retropropagacao <- function(arq,dados,n,limiar){
	erroQuadratico <- c(2*limiar,2*limiar,2*limiar)
	epocas <- 0

	# >>> DIFERENTE <<<
	# Treina enquanto erroQuadratico > limiar
	sao.maiores <- TRUE 
	while(sao.maiores){
		erroQuadratico <- 0
		i <-0
		# Treino para todos os exemplos
		for(i in 1:nrow(dados)){
			# Pego um exemplo de entrada (treino)
			x.entrada<- dados[i,1:arq$num.entrada] # primeiras duas colunas

			# >>> DIFERENTE <<<
			# agora, a ultima coluna da base de dados contém strings
			# que precisam ser tratadas para utilização matemática
			x.saida <- arq$desclassificador(dados[i,ncol(dados)]) # ultima coluna

			# Saida da rede para o exemplo
			resultado <- mlp.propagacao(arq,x.entrada)
			y <- (resultado$y.escondida.saida)

			# Erro
			erro <- x.saida - y
			erroQuadratico <- erroQuadratico + erro*erro

			# Gradiente local do neuronio de saida
			# erro * derivada da funcao de ativacao
			grad.local.saida <- erro*arq$der.funcao.ativacao(y)

			# >>> DIFERENTE <<<
			grad.local.saida <- t(grad.local.saida) # grad.local.escondida agora é
													# uma matriz

			# Gradiente local dos neuronios escondidos
			# derivada da funcao de ativacao * (somatorio dos gradientes locais * pesos)
			pesos.saida <- arq$saida[,1:arq$num.escondida]
			grad.local.escondida <- as.numeric(arq$der.funcao.ativacao(resultado$y.entrada.escondida))*(grad.local.saida %*% pesos.saida)

			# Ajuste dos pesos
			# Saida
			arq$saida <- arq$saida + n * (t(grad.local.saida)	 %*% c(resultado$y.entrada.escondida,1))

			# Escondida

			# >>> DIFERENTE <<<
			# grad.local.escondida é uma matriz
			arq$escondida <- arq$escondida+n*(t(grad.local.escondida)%*%as.numeric(c(x.entrada,1))) # bias
		}
		erroQuadratico <- erroQuadratico / nrow(dados)
		#cat("Erro Quadratico Medio = ", erroQuadratico, "\n")
		epocas <- epocas+1

		# >>> DIFERENTE <<<
		# Loop: verificar se todas as saídas ainda são são maiores que o limiar
		todas <- FALSE
		for(e in erroQuadratico){
			todas <- todas | (e > limiar)
		}
		sao.maiores <- sao.maiores & todas

	}
	retorno <- list()
	retorno$arq <- arq
	retorno$epocas <- epocas
	return(retorno)
}

# >>> DIFERENTE <<<
# classificaIris
# Objetivo: retornar a string correspondente à classe (na base de dados iris) da saída
classificaIris <- function(resultado){
	if(resultado$y.escondida.saida[1] > resultado$y.escondida.saida[2] 
		& resultado$y.escondida.saida[1] > resultado$y.escondida.saida[3]) return('setosa')
	if(resultado$y.escondida.saida[2] > resultado$y.escondida.saida[1] 
		& resultado$y.escondida.saida[2] > resultado$y.escondida.saida[3]) return('versicolor')
	if(resultado$y.escondida.saida[3] > resultado$y.escondida.saida[2] 
		& resultado$y.escondida.saida[3] > resultado$y.escondida.saida[1]) return('virginica')
} # fim classificaIris

# main
dados <- iris # base de dados iris

# >>> DIFERENTE <<<
dados[,1:4] <- scale(dados[,1:3], scale=FALSE) 	# heurística de normalizaçao
												# para os dados numéricos do dataframe

arq <- arquitetura(4,8,3,funcao.ativacao,der.funcao.ativacao,desclassificadorIsis) # >>> DIFERENTE <<<

for(i in 1:10){        
    a <- as.integer(i* 15 + 1)
    b <- as.integer(1 + (i -1) * 15)
    c<- as.integer(i* 15)
    d <- as.integer((i -1) * 15)
    teste <- dados[b:c,]
    if(i != 10 && i != 1){
        treinamento <- dados[1: d,]
        treinamento <- rbind(treinamento,dados[a : 150,])
    }else{
        if(i == 1)
            treinamento <- dados[a: 150,]
        else
            treinamento <- dados[1: d,]
    }

    X <- treinamento[,1:4]
    Y <- treinamento[,5]
    
	modelo <- mlp.retropropagacao(arq,X,.1,1e-2)
	cat("Iteracao", i, "concluida!\nErro:")

    X.out <- teste[,1:4]
    Y.out <- teste[,5]
    pred <- mlp.propagacao(modelo, X.out)
    erro <- sum(pred != Y.out)/15
    cat(erro, "\n")
}



# Arquitetura:
# 4 neuronios na camada de entrada, pois há 4 atributos (com excessão da classe) em cada instância da base de dados
# # 8 neuronios na camada escondida: trata-se de uma fração razoavel 
# #   da quantidade total de dados (150) na base de dados (dados)
# # 3 neuronios na camada de saída, dada a expecificação do trabalho

# # >>> DIFERENTE <<<
# # teste (após a normlização):
# print(classificaIris(mlp.propagacao(modelo$arq, c(-0.74333333, 0.44266667, -2.358, 0.2)))) # é pra ser setosa
# print(classificaIris(mlp.propagacao(modelo$arq, c(1.15666667, 0.14266667, 0.942, 1.4)))) # é pra ser versicolor
# print(classificaIris(mlp.propagacao(modelo$arq, c(0.45666667, 0.24266667, 2.242, 2.5)))) # é pra ser virginica