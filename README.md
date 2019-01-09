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
