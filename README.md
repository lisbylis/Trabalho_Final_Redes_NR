# Prevendo a Magnitude de Tornados nos EUA utilizando Redes Neurais
- Autores do projeto: Anelise Gonçalves Silva, Gustavo Uchoa Barros e Matheus Zaia Monteiro

## Descrição
- Esse é o trabalho final da disciplina de Redes Neurais e Algoritmos Genéticos que tem como objetivo realizar a análise dos dados do dataset [tornado.cvs](https://github.com/lisbylis/Trabalho_Final_Redes_NR/blob/main/tornado.csv) e buscar prever qual seria a magnitude de um determinado tornado através dos atributos selecionados.

# Introdução Teórica 

## Tornados 
- Um tornado é um fenômeno meteorológico que se manifesta como uma coluna de ar que gira de forma violenta e potencialmente perigosa, estando em contato tanto com a superfície da Terra como com uma nuvem de chuva. A maioria dos tornados conta com ventos que chegam a velocidades entre 65 e 180 quilômetros por hora, mede aproximadamente 75 metros de diâmetro e translada-se por vários metros, senão quilômetros, antes de desaparecer. Os mais extremos podem ter ventos com velocidades superiores a 480 km/h, medir até 1 500 m de diâmetro e permanecer no solo, percorrendo mais de 100 km de distância. 
- Os tornados são observados em todos os continentes, exceto na Antártida. No entanto, a maioria dos tornados no mundo ocorre no "Corredor dos Tornados", uma região dos Estados Unidos, embora possam ocorrer quase em qualquer lugar na América do Norte. Segundo a Agência Oceânica e Atmosférica Americana (NOAA) e o Serviço Meteorológico Nacional, responsáveis por medir a intensidade, monitorar e emitir os alertas à população sobre as tempestades, são cerca de 1,2 mil tornados por ano. A incidência de tornados nos EUA geralmente dá-se no centro-oeste do país e próximo às Montanhas Rochosas, durante os meses de abril a setembro.

## Por que os EUA são tão atingidos por tornados?
- Os cientistas dizem que as causas que levam à formação de tornados ainda não são completamente entendidas. Mas sabe-se que eles tendem a ocorrer quando o ar frio e seco colide com o ar quente e úmido. Isso acontece com mais frequência em latitudes médias, exatamente onde ficam os EUA. Além disso, há o ar frio que flui livremente do norte através das Grandes Planícies, a proximidade das águas quentes do Golfo do México e o ar seco vindo das Montanhas Rochosas.

## Como eles são medidos e detectados?
- Os tornados podem ser detectados através de radares de impulsos Doppler, assim como visualmente, por caçadores de tempestades.
- A sua intensidade é medida através da escala Fujita, porém, a partir de Fevereiro de 2007, foi adotada a Escala Fujita Melhorada. A escala tem o mesmo design básico da escala Fujita original: seis categorias de zero a cinco representando graus crescentes de danos. Ela foi revista para permitir melhores avaliações de levantamentos dos danos dos tornados, de modo a alinhar as velocidades do vento mais estreitamente com danos associados à tempestade. 
Padronizando e elucidando melhor o que antes era subjetivo e ambíguo, também acrescenta mais tipos de estruturas e vegetação, amplia graus de dano e explica melhor variáveis como diferenças na qualidade da construção. Uma categoria "EF-Unknown" (EFU) foi posteriormente adicionada para tornados que não podem ser classificados devido à falta de evidências de danos.

## Escala EF 
- EFU : Velocidade desconhecida - Sem danos detectáveis
- EF0 : Velocidade entre 65 e 86mph - Danos Leves
- EF1 : Velocidade entre 86 e 110mph - Danos Moderados
- EF2 : Velocidade entre 111 e 135mph - Danos Consideráveis
- EF3 : Velocidade entre 136 e 165mph - Danos Graves
- EF4 : Velocidade entre 166 e 200mph - Danos Devastadores
- EF5 : Velocidade maior que 200mph - Danos Catastróficos
  
# Por dentro do Código
## Atributos utilizados 
Mo: Mês (1 - 12)

Slat: Latitude inicial em Graus decimais (uma alternativa ao uso de graus, minutos e segundos).

Slon: Longitude inicial em Graus decimais.

Len: Comprimento em milhas

Wid: Largura do tornado em jardas 

## Target
Mn: Magnitude do Tornado (medida na escala Fujita até Janeiro de 2007 e a partir disso, medida na escala Fujita Aprimorada)

## Como me orientar no git?
- O conjunto de dados referentes ao nosso problema é representado pelo arquivo [tornado.cvs](https://github.com/lisbylis/Trabalho_Final_Redes_NR/blob/main/tornado.csv) .
- O arquivo [classes.py](https://github.com/lisbylis/Trabalho_Final_Redes_NR/blob/main/classes.py) contém todas as classes e funções que utilizamos no notebook principal para montar a nossa rede neural.
- Também deixamos no git, os jobs e scripts utilizados para rodar de fato a nossa rede neural, que podem ser encontrados nos arquivos: [job_gpu.sh](https://github.com/lisbylis/Trabalho_Final_Redes_NR/blob/main/job_gpu.sh) , [trabalho_final_optuna_gpu.py](https://github.com/lisbylis/Trabalho_Final_Redes_NR/blob/main/trabalho_final_optuna_gpu.py)
- Toda a explicação de como foi utilizado esses arquivos está no próprio notebook final que contem os resultados e discussões.

# Hiperparâmetros
Como sabemos, os hiperparâmetros são escolhidos antes do treinamento, durante a concepção da estrutura e comportamento do modelo. Dentre eles, se incluem o número de camadas de uma rede, a taxa de aprendizado, o otimizador usado para convergir uma determinada função de perda, o número de épocas, tamanho dos lotes e assim por diante.

## Arquitetura da Rede Neural
Foram utilizadas, no total, cerca de 100 arquiteturas.
- Camadas Ocultas: de 1 a 3 camadas
- Número de Neurônios por camada oculta: 3 a 16 neurônios

## Funções de Ativação utilizadas
As funções de ativação são importantíssimas no desenvolvimento de uma rede neural. São elas que vão ser responsáveis por transmitir a informação através da combinação de pesos e entradas. Para o nosso modelo, utilizamos as seguintes:
- ReLU: Introduz a propriedade de não linearidade em um modelo de deep learning e resolve o problema dos gradientes que desaparecem. Para o cálculo do backpropagation de redes neurais, a diferenciação para o ReLU é relativamente fácil. A única suposição que faremos é a derivada no ponto zero, que também será considerada zero.
- Sigmoide: A função de ativação sigmoide é comumente utilizada por redes neurais com propagação positiva (Feedforward) que precisam ter como saída apenas números positivos, em redes neurais multicamadas e em outras redes com sinais contínuos.
- Tangente Hiperbólico: A função de ativação tangente hiperbólica possui uso muito comum em redes neurais cujas saídas devem ser entre -1 e 1.
- Leaky ReLU: Na função ReLU, o gradiente é 0 para x < 0, o que fez os neurônios morrerem por ativações nessa região. Leaky ReLU ajuda a resolver este problema. Em vez de definir a função Relu como 0 para x inferior a 0, definimos como um pequeno componente linear de x.

## Otimizadores
Os otimizadores vão atuar depois do backpropagation, quando inicia-se o ajuste dos pesos e viéses da rede neural com o objetivo de minimizar o erro na rede neural. No presente trabalho, utilizamos 3:
- Descida do gradiente estocástico (SGD): Ao invés de atualizar os parâmetros do modelo após uma época, o SGD atualiza os pesos após cada batch.
- RMSPROP e Adam: permitem com que a taxa de aprendizado seja atualizada conforme os dados são processados pela rede. Quando o gradiente encontrado é pequeno, a taxa de aprendizado assume um valor maior, passando mais rapidamente por regiões planas. Quando um gradiente grande é encontrado, indicando uma região mais inclinada no hiperplano de parâmetros, a taxa de aprendizado é reduzida de forma a convergir para a região de mínimo. O Adam nesse caso é bem parecido com o RMSPROP, porém utiliza algumas modificações e combinações de técnicas de outros otimizadores: como o momento e velocidade.

## Optuna
- O Optuna é uma biblioteca Python robusta e de código aberto desenvolvida para simplificar a otimização de hiperparâmetros em aprendizado de máquina. Ele oferece uma interface intuitiva para otimizar hiperparâmetros, permitindo explorar com eficiência o vasto espaço de pesquisa e determinar a configuração ideal para o seu modelo. A principal funcionalidade do Optuna reside na capacidade de pesquisar automaticamente os melhores hiperparâmetros por meio de vários algoritmos de otimização e estratégias de remoção.

# Referências:
[1] . https://www.kaggle.com/datasets/danbraswell/us-tornado-dataset-1950-2021 (dataset)

[2] . https://pt.wikipedia.org/wiki/Tornado (o que são tornados?)

[3] . https://www.bbc.com/portuguese/articles/crgqrz98d9lo (por que os Estados Unidos são tão afetados por tornados?)

[4] . https://pt.wikipedia.org/wiki/Escala_Fujita_melhorada  (Escala Fujita Melhorada, o que é?)

[5]. Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam). Link: https://www.youtube.com/watch?v=NE88eqLngkg

[6]. https://www.hackersrealm.net/post/normalize-data-using-max-absolute-min-max-scaling (Normalização por Máximo Absoluto)

[7]. https://builtin.com/machine-learning/relu-activation-function (Função de Ativação ReLU)

[8]. https://www.deeplearningbook.com.br/funcao-de-ativacao/ (funções de ativação)

[9]. https://medium.com/data-and-beyond/master-the-power-of-optuna-a-step-by-step-guide-ed43500e9b95 (Optuna)

