# Notes for training
Note per confrontare le prove di training su diversi dataset e configurazioni.

## Oxford
Note generali
- `train_file_generator.py`
1. Rimozione frames sovraesposti (normalmente primi N=100 frames di ogni dataset)
2. Rimozione ultimi M=100 frames dove normalmente la macchina è ferma
3. Viene creato un file **files.txt** che contiene tutti i frames che vengono considerati
del dataset
4. Divisione train e test in rispettivamente 90% e 10% di **files.txt**


### 2014-11-18-13-20-12 
Main route.

- Loss oscilla tra 0.29 e 0.35 anche dopo 90 000 steps;
- Test su immagine campione porta ad una depth molto sgranata.

### 2014-05-06-12-54-54
Alterate route.

- Risultati migliori rispetto a 2014-11-18-13-20-12;
- Test su immagine campione porta ad una depth sgranata ma che si capisce.

### 2014-11-28-12-07-13
Main route.


### Multi dataset
Utilizzo di più routes.
Idea:

- Considero diverse alterate-routes (inizio con 3);

Dataset considerati:

- 2014-05-06-12-54-54
- 2014-05-14-13-46-12
- 2014-05-19-12-51-39


## TO DO
1. Prova con dimensione diversa dei files in formato struct2depth
2. Aumentare numero di steps
3. Provare differenti configurazioni della rete