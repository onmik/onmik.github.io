import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import timeit
import math
import time

img = tif.imread('linear.tif')
#img = tif.imread('stacked.tif')
print(img.shape)

imglst = img.tolist()
"""
#----------------------------------- asinh ------------------------------------ 
# pixel = (original - blackpoint) * asinh(original * stretch) / original * asinh(stretch)
n = 100
np_single_time = timeit.timeit(stmt='((1 - 0.1) * np.arcsinh(1 * 500)) / (1 * np.arcsinh(500))', globals=globals(), number=n ) / n

math_single_time = timeit.timeit(stmt='((1 - 0.1) * math.asinh(1 * 500)) / (1 * math.asinh(500))', globals=globals(), number=n) / n

def compare(a, b):
    def get_name(var):
        for name, value in globals().items():
            if value is var:
                return name
    print("\n")
    if round(a, 2) > 0:
        print(get_name((a)), "execution time is", round(a, 2), "seconds")
    else:
        print(get_name((a)), "execution time is", a, "seconds")
    if round(b, 2) > 0:
        print(get_name((b)), "execution time is", round(b, 2), "seconds\n")
    else:
        print(get_name((b)), "execution time is", b, "seconds\n")

    print(get_name(min(a,b)), "is", round(max(a,b) / min(a,b), 2), "times faster than", get_name(max(a,b)), "\n")
    
compare(np_single_time, math_single_time)

def arcsinh(image, black, stretch):
    return  ((image - black) * np.arcsinh(image * stretch)) / (image * np.arcsinh(stretch))


#plt.imshow(pixel)

#plt.imshow(pixel, cmap='gray')

def asinh(image, black, stretch):
    for x in range(len(imglst)):
        for y in range(len(imglst[0])):
            image[x][y] = (image[x][y] - black) * (math.asinh(image[x][y] * stretch)) / (image[x][y] * math.asinh(stretch))

    return image

start = time.time()
np_matrix = arcsinh((arcsinh((arcsinh(img, 0.003, 1000)), 0.10, 100)), 0.15, 30)
end = time.time()
np_matrix_time = end - start

start = time.time()
math_matrix = asinh((asinh((asinh(imglst, 0.003, 1000)), 0.10, 100)), 0.15, 30)
end = time.time()
math_matrix_time = end - start
print(math_matrix_time)

compare(np_matrix_time, math_matrix_time)

tif.imwrite('np_matrix.tif', np_matrix)
tif.imwrite('math_matrix.tif', math_matrix)


#----------------------------- MTF --------------------------------------------
def mtf(image, m, h, s):
    xp = (image - s) / (h - s)
    return (m - 1) * xp / (2 * m - 1) * xp - m
    
mtf_matrix = mtf(img, )


xp = (img - 0.1) / (1 - 0.1)
mt = ((0.4 - 1) * xp) / ((2 * 0.4 - 1) * xp - 0.4)

mtf = np.array(xp)
for x in range(mtf.shape[0]):
    for y in range(mtf.shape[1]):
        mtf[x][y] = ((0.4 - 1) * mtf[x][y]) / ((2 * 0.4 - 1) * mtf[x][y] - 0.4)
"""        
        
#print(mt-mtf)

a = np.arange(10)





