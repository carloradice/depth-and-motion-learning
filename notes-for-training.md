# Notes for training
Note per confrontare le prove di training su diversi dataset e configurazioni.

## Oxford

### 2014-11-18-13-20-12 
Main-route.

- Loss oscilla tra 0.29 e 0.35 anche dopo 90 000 steps;
- Test su immagine campione porta ad una depth molto sgranata.

### 2014-05-06-12-54-54
Alterate-route.

- Risultati migliori rispetto a 2014-11-18-13-20-12;
- Test su immagine campione porta ad una depth sgranata ma che si capisce.

### Multi dataset
Utilizzo di più routes.
Idea:

- Considero diverse alterate-routes (inizio con 3);
- Rimozione frames sovraesposti (normalmente primi N frames di ogni dataset).

Dataset considerati:

- 2014-05-06-12-54-54
- 2014-05-14-13-46-12
- 2014-05-19-12-51-39
