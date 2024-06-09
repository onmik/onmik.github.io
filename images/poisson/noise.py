import matplotlib.pyplot as plt
import numpy as np


#------------------- poisson example-----------------
pixel_small = 100
pixel_big = pixel_small * 4

def poisson_1d(pixel):
    sample = np.random.poisson(pixel, 5000)
    dev = np.std(sample)
    print("st. odchylka: ",  str(dev))
    
    bin = np.arange(0, 500, 1)
    plt.hist(sample, bins=bin) 
    plt.xlabel("photon count")
    plt.ylabel("probability") 
    plt.title("st. odchylka: " + str(dev))
    plt.show()
    plt.savefig(str(pixel) + ".jpg")
    plt.close()

poisson_1d(pixel_small)
poisson_1d(pixel_big)    
"""
#-----------------photon noise--------------------------
def generate_img(resolution):
    x, y = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution))
    return ((0.5 + x + x**10 + y/4 + y**8) * np.exp(-x**1.8 - y**1.5) * 250).astype(int)
    
def resample(img, binning):
    big = img.reshape(img.shape[0]//binning, binning, img.shape[1]//binning, binning)
    return big.sum(axis=(1, 3))

def poisson(img):
    return np.random.poisson(img, (img.shape[0], img.shape[1]))

def figure(fig, name):
    plt.imshow(fig, cmap='gray')
    plt.colorbar()
    plt.title(name)
    plt.savefig(name + ".jpg" , dpi=600)
    plt.close()
    
# ------------------ Noiseless----------------   
noiseless = generate_img(3000)
figure(noiseless, "noiseless")
print("Full resolution illumination: ", noiseless.sum())

noiseless_resampled_2 = resample(noiseless, 2)
#figure(noiseless_resampled_2, "noiseless_resampled_2")
print("Resampled 2 illumination: ", noiseless_resampled_2.sum())

noiseless_resampled_5 = resample(noiseless, 5)
#figure(noiseless_resampled_5, "noiseless_resampled_5")
print("Resampled 5 illumination: ", noiseless_resampled_5.sum())    

#------------------ with noise---------------------------------
noise = poisson(noiseless)
figure(noise, "full_resolution")

noise_resampled_2 = poisson(noiseless_resampled_2)
figure(noise_resampled_2, "resampled_2")

noise_resampled_5 = poisson(noiseless_resampled_5)
figure(noise_resampled_5, "resampled_5")


"""