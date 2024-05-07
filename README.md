# Prevendo a Magnitude de Tornados nos EUA utilizando Redes Neurais
- Autores do projeto: Anelise Gonçalves Silva, Gustavo Uchoa Barros e Matheus Zaia Monteiro

## Descrição
- Esse é o trabalho final da disciplina de Redes Neurais e Algoritmos Genéticos que tem como objetivo realizar a análise dos dados do dataset 'Us_Tornado_Dataset' [1] e buscar prever qual seria a magnitude de um determinado tornado através dos atributos selecionados.

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
Mn : Mês (1 - 12)

Slat: Latitude inicial em Graus decimais (uma alternativa ao uso de graus, minutos e segundos).

Slon: Longitude inicial em Graus decimais.

Len: Comprimento em milhas

Wid: Largura em jardas 

## Target
Mn: Magnitude do Tornado (medida na escala Fujita até Janeiro de 2007 e a partir disso, medida na escala Fujita Aprimorada)





# Referências:
[1] . https://www.kaggle.com/datasets/danbraswell/us-tornado-dataset-1950-2021 (dataset)
[2] . https://pt.wikipedia.org/wiki/Tornado (o que são tornados?)
[3] . https://www.bbc.com/portuguese/articles/crgqrz98d9lo (por que os Estados Unidos são tão afetados por tornados?)
[4] . https://pt.wikipedia.org/wiki/Escala_Fujita_melhorada  (Escala Fujita Melhorada, o que é?)



