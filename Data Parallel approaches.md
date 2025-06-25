# Data Parallel approaches





# 1) Vanilla appraoch, no overlapping of computation and communication

![](C:\Users\manu.chauhan\Downloads\tiny-huggingface-accelerate\assets\image-20250625000622418.png)



# 2) async reduce op : overlap, but computation for all gradients individually instead in a group

![](C:\Users\manu.chauhan\Downloads\tiny-huggingface-accelerate\assets\image-20250625000649699.png)



# 3) Bucket the gradients to improve utilization 

![image-20250625000713909](C:\Users\manu.chauhan\Downloads\tiny-huggingface-accelerate\assets\image-20250625000713909.png)