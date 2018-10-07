# Knapsack

### Given:

 - capacity K
 - n items
 - value ![V_i](https://latex.codecogs.com/gif.latex?%5Cinline%20V_i)
 - weight ![W_i](https://latex.codecogs.com/gif.latex?%5Cinline%20W_i)
 - ![X_i](https://latex.codecogs.com/gif.latex?%5Cinline%20X_i) take item i

### Goal:

Find the greatest total value such that total weight does not exceeds capacity of knapsack:

maximize:

![max_func](https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%5C%20%5Cin%5C%201%5C%2C...%5C%2Cn%7D%5C%20v_i%5C%2Cx_i)

subject to:

![sum_constraint](https://latex.codecogs.com/gif.latex?%5Cinline%20c_i%5C%2C%5Cne%5C%2Cc_j%20%5C%20%28%3Ci%2Cj%3E%5C%20%5Cin%5C%20E%29)  
![x_i_constraint](https://latex.codecogs.com/gif.latex?x_i%5C%20%5Cin%5C%20%5C%7B0%2C%5C%2C1%5C%7D%5C%20%28i%5C%20%5Cin%5C%201%5C%2C...%5C%2Cn%29)
