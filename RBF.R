
# Implementacao simples da RBF
# http://www.di.fc.ul.pt/~jpn/r/rbf/rbf.html

# Cria um dataframe com 2 colunas, 100 observacoes
# Esses serao os dados de treinamento

#    N <- 300
#    X <- data.frame(x1=runif(N, min=-1, max=1),
#    x2=runif(N, min=-1, max=1))

# Cria os valores de saida
#   Y <- target(X$x1, X$x2)

# Plota o grafico
#    plot(X$x1, X$x2, col=Y+3)



# Funcao RBF com o LSM com pseudo-inversa
# Retorna um modelo RBF dado:
# * observacoes x1...xN do dataset
# * valor de saida de cada observacao
# * numero de centros
# * valor gama para a funcao Gaussiana
# Precisa so pacote corpcor para a pseudo inversa

    library(corpcor)
    rbf <- function(X, Y, K=10, gama=1.0) {
        N <- dim(X)[1] # numero de observacoes
        ncols <- dim(X)[2] # numero de variaveis
        repeat {
            km <- kmeans(X, K) # agrupa os dados em K grupos
            if (min(km$size)>0) # nao pode haver grupos vazios
            break
        }
        
        mus <- km$centers # centros dos grupos (medias)

    # Calcula as saidas das Gaussianas
        Phi <- matrix(rep(NA,(K+1)*N), ncol=K+1) # Vai armazenar todas as saidas mais o bias
        for (lin in 1:N) {
        
            Phi[lin,1] <- 1
            # coluna do bias
            for (col in 1:K) {
                Phi[lin,col+1] <- exp( (-1/(2*gama*gama)) * sum((X[lin,]-mus[col,])*(X[lin,]-mus[col,])) )

                #Phi[lin,col+1] <- exp( -gama * sum((X[lin,]-mus[col,])*(X[lin,]-mus[col,])))
                #Phi[lin,col+1] <- exp( -gama * norm(as.matrix(X[lin,]-mus[col,]),"F")^2 )
            }
        }
        
        # Calcula os pesos com a pseudo inversa -> w = inversa(t(Phi) * Phi) * t(Phi) * Y
        # Encontra os pesos fazendo a inversa
        # %*% é para multiplicacao de matrizes
        w <- pseudoinverse(t(Phi) %*% Phi) %*% t(Phi) %*% Y
    
        return(list(pesos=w, centros=mus, gama=gama))
}
 # retorna o modelo RBF
# Treina o modelo
modelo <- rbf(X, Y) # using default values for K and gamma
modelo

# Implementacao da funcao para predicao
    rbf.predict <- function(modelo, X, classification=FALSE) {
        gama <- modelo$gama
        centros <- modelo$centros
        w <-  modelo$pesos
        N <-  dim(X)[1]

        # numero de observacoes
        pred <- rep(w[1],N)
        # inicia com o peso do bias ja que a entrada associada eh 1
        for (j in 1:N) {
            # Predicao para o ponto xj
            for (k in 1:length(centros[,1])) {
                # o peso para o centro[k] é dado por w[k+1] porque w[1] eh o bias
                pred[j] <- pred[j] + w[k+1] * exp( (-1/(2*gama*gama)) *
                sum((X[j,]-centros[k,])*(X[j,]-centros[k,])) )
            #pred[j]<-pred[j]+w[k+1]*exp(-gama*sum((X[j,]-centros[k,])*(X[j,]-centros[k,])))
            #pred[j]<-pred[j]+w[k+1]*exp(-gama*norm(as.matrix(X[j,]-centros[k,]),"F")^2)
            }
        }
        
        # Se for classificacao, aplica a funcao sinal em cada pred
        if (classification) {
        pred <- unlist(lapply(pred, sign))
        }
        return(pred)
    }
    
    target <- function(x1, x2) {
        2*(x2 - x1 + .25*sin(pi*x1) >= 0)-1
    }


#Cria os dados teste
    N.test <- 200
    X.out <- data.frame(x1=runif(N.test, min=-1, max=1),
    x2=runif(N.test, min=-1, max=1))
    Y.out <- target(X.out$x1, X.out$x2)

    # Faz predicao nos dados de teste
    rbf.pred <- rbf.predict(modelo, X.out, classification=TRUE)
    # Verifica o erro
    erro <- sum(rbf.pred != Y.out)/N.test
    erro

    # Mostrando os resultados graficamente
    plot(X.out$x1, X.out$x2, col=Y.out+3, pch=0)
    points(X.out$x1, X.out$x2, col=rbf.pred+3, pch=3)
    points(modelo$centros, col="black", pch=19) # draw the model centers
    legend("topleft",c("true value","predicted"),pch=c(0,3),bg="white")
    target <- function(x1, x2) {
        2*(x2 - x1 + .25*sin(pi*x1) >= 0)-1
    }
        
