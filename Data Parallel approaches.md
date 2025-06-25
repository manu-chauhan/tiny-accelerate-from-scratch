# Data Parallel approaches





# 1) Vanilla appraoch, no overlapping of computation and communication

![](assets\image-20250625000622418.png)



# 2) async reduce op : overlap, but computation for all gradients individually instead in a group

![](assets\image-20250625000649699.png)



# 3) Bucket the gradients to improve utilization 

![image-20250625000713909](assets\image-20250625000713909.png)