# Knapsack

### Given:

 - capacity K
 - n items
 - value ![V_i](https://latex.codecogs.com/gif.latex?%5Cinline%20V_i)
 - weight ![W_i](https://latex.codecogs.com/gif.latex?%5Cinline%20W_i)
 - ![X_i](https://latex.codecogs.com/gif.latex?%5Cinline%20X_i) take item i

### Goal:

Find the greatest total value such that total weight does not exceed capacity of the knapsack.

Maximize:

![max_func](https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%5C%20%5Cin%5C%201%5C%2C...%5C%2Cn%7D%5C%20v_i%5C%2Cx_i)

subject to:

![sum_constraint](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Csum_%7Bi%20%5Cin%201%5C%2C...%5C%2Cn%7D%20w_i%20%5Ccdot%20x_i%20%5Cle%20K)  
![x_i_constraint](https://latex.codecogs.com/gif.latex?x_i%5C%20%5Cin%5C%20%5C%7B0%2C%5C%2C1%5C%7D%5C%20%28i%5C%20%5Cin%5C%201%5C%2C...%5C%2Cn%29)

# Benchmark

|Name|Capacity|Found value|Time (sec)|
|----:|----:|----:|----:|
|4_0|11|19|0.05|
|19_0|31181|12248|0.14|
|30_0|100000|99798|0.23|
|40_0|100000|99924|0.32|
|45_0|58181|23974|0.4|
|50_0|341045|142156|0.52|
|50_1|5000|5345|0.5|
|60_0|100000|99837|0.52|
|82_0|104723596|104723596|53.95|
|100_0|100000|99837|0.94|
|100_1|3190802|1333930|2.01|
|100_2|10000|10892|1.2|
|106_0|106925262|106925262|982.96|
|200_0|100000|100236|1.76|
|200_1|2640230|1103604|38.12|
|300_0|4040184|1688692|134.67|
|400_0|9486367|3967180|376.32|
|500_0|50000|54939|21.06|
|1000_0|100000|109899|109.51|
|10000_0|1000000|1099893|8075.07|
