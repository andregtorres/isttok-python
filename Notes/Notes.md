Caros,

Segue a correção para o campo vertical, que não tinha colocado bem os sentido das correntes,

## Vertical Coils: 4 coils, 5 turns, R1,2=58 [cm],R2,3=35 [cm],z=±7 [cm]

Bpoloidal:
[-2.319e-05 -1.970e-05 -2.749e-06  4.486e-06  2.142e-05  1.890e-05
  1.890e-05  2.142e-05  4.486e-06 -2.749e-06 -1.970e-05 -2.319e-05]
Bradial:
[-5.610e-07  2.409e-05  2.313e-05  2.229e-05  2.312e-05 -4.342e-06
  4.342e-06 -2.312e-05 -2.229e-05 -2.313e-05 -2.409e-05  5.610e-07]
BR:
[ 1.452e-07  1.772e-08 -1.697e-07 -4.440e-07  3.707e-07  3.503e-06
  1.844e-06  1.104e-06  7.306e-07  5.110e-07  3.648e-07  2.518e-07]
BZ:
[-3.773e-07 -5.057e-07 -7.435e-07 -1.352e-06 -3.420e-06 -7.041e-07
  2.385e-07  1.198e-07 -2.388e-08 -1.307e-07 -2.132e-07 -2.892e-07]
  
 Polarity -1 -1 1 -1  1 1 1 1  1 1 -1 1 1  
## Horizontal Coils: 2 coils , 4 turns, R1,2=58 [cm],z=±7[cm]

Bpoloidal :
[-1.11184244e-05, -1.43062868e-05, -3.53365432e-06, -8.75840081e-07, -9.48056345e-08,  3.57557241e-08,
 -3.57557241e-08,  9.48056345e-08, 8.75840081e-07,  3.53365432e-06,  1.43062868e-05,  1.11184244e-05]
Bradial: 
[-1.78602907e-05,  5.70732275e-06,  6.16853719e-06,  4.13872674e-06, 2.96554276e-06,  2.38941947e-06,  
2.38941947e-06,  2.96554276e-06, 4.13872674e-06,  6.16853719e-06,  5.70732275e-06, -1.78602907e-05])
BZ:
[-6.11698989e-06, -1.41517590e-05, -6.87292642e-06, -3.77101895e-06, -2.02991769e-06, -6.52964643e-07,  
6.52964643e-07,  2.02991769e-06, 3.77101895e-06,  6.87292642e-06,  1.41517590e-05,  6.11698989e-06]
BR:
[-2.01293760e-05, -6.08038578e-06, -1.81671306e-06, -1.91717786e-06, -2.16399310e-06, -2.29874772e-06, 
-2.29874772e-06, -2.16399310e-06, -1.91717786e-06, -1.81671306e-06, -6.08038578e-06, -2.01293760e-05]

# Setembr Toquim mudou os ganhos. Trocou o andar de entrada, Filtro 1nF 
  
### 28/04/2018 21:5

Viva,

Já tenho o meu modelo de “Espaço de Estados” do ISTTOK a funcionar (sem plasma)
Segue uma simulação com um modelo de 6 ‘condutores’ na copper shell onde se vê o campo provocado por
estas correntes na sondas de Mirnov, por resposta a um Heaviside nas coils de campo vertical (step response do sistema (A,B,C,D))

Acho que tem resultados interessantes, nomeadamente em alguns sinais (e.g. ‘m3’ ) que se afastam claramente da
exponencial do modelo simples de passa-baixo. 
Penso que podemos discutir isto melhor, pois há alguma parametrização a fazer e claro uma 
optimização com os resultados experimentais.
Nesta altura a soma de correntes no cobre ainda não é nula, como tem de ser pelo facto do copper shell não ser fechado toroidalmente, mas tenho já ideia de como o fazer.

## Andre Shots
shotV=42952 Ipfc = 340A
shotH=42966 Ipfc = 260A
shotP=43066

## 28/09/2018 
Shots Horizontal
Humberto
Num 44267 até #44272
44267 slopes
[ 0.0097 -0.0602  0.002  -0.0103 -0.1549 -0.1806 
-0.1022 -0.1603 -0.1589  0.0072 -0.2403 -0.16  ]
  
##30 out '18
44473 Vert +300A
44475 Vert -300A
44476 Hori +250A c/ mirnov12 descomp
44480 Hori +200A
44481 Hori -200A
44499 Prim com resistência externa
44501 Prim +150A
44503 Prim -150A ferro saturado
44505 3 negativos desfasados 50ms
44507 3 positivos "  "


Shots Vertical #44278

44330 - Horizontal,

[-0.0452 -0.1018 -0.0039 -0.0022 -0.1448 -0.2128 
-0.1636 -0.1808 -0.1323  0.0107 -0.2344 -0.1615]
  
Shot Plasma # #44281

 Nshot=44355 # Plasma 30 Pulses

Nshot=44360 # Plasma 10 Pulses
Hugo 44354 foi o melhorzinho mas se calhar preferes os que vêm a seguir até ao 44364 por serem mais parecidos entre si.
T=-1000 e - 900 são o WO usado como correcção, de - 800 a -300 são 6 WOs em 6 segundos consecutivos para se ter uma ideia da variação 

BiotSart form Andre
H2
Out[135]: 
array([ 1.45132498e-08,  1.59595068e-07,  1.24699365e-08,  2.11370544e-10,
       -2.11251399e-09, -2.24109804e-09, -1.90269723e-09, -3.09836179e-09,
       -1.02211276e-08, -4.52153503e-08, -4.51707203e-08, -7.60930718e-09])
       