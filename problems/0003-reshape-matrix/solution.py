import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method

	if len(a)*len(a[0])!=new_shape[0]*new_shape[1]:
		return []

	temp=[]
	for num in a:
		temp.extend(num)
		
	ans=[]
	temp1=[]
	for ele in temp:
		temp1.append(ele)
		if len(temp1)==new_shape[1]:
			ans.append(temp1)
			temp1=[]
	


	return ans