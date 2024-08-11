import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
from PIL import Image

img = np.asarray(Image.open('linear.tif'))

class Stretch:
    def __init__(self, image):
        self.image = image
        
    def AsinhStretch(self, black, stretch):
        if stretch == 0:
            out = self.image
        else:
            out = ((self.image - black) * np.arcsinh(self.image * stretch)) / (self.image * np.arcsinh(stretch))
        return out
   
    def plot_asinh(self, s=0, b=0):
        fig, (ax1, ax2) = plt.subplots(2, height_ratios=[2, 0.5])
        ax1.imshow(self.image, vmin=0, vmax=1, cmap='gray')
        ax2.hist(self.image.ravel(), 256, (0, 1))

        fig.subplots_adjust(bottom=0.25)

        axstretch = fig.add_axes([0.2, 0.15, 0.6, 0.02])
        stretch_slider = Slider(
            ax=axstretch,
            label="stretch ",
            valmin=0,
            valmax=1000,
            valinit=s
        )

        axblack = fig.add_axes([0.2, 0.1, 0.6, 0.02])
        black_slider = Slider(
            ax=axblack,
            label="b ",
            valmin=0,
            valmax=0.2,
            valinit=b
        )
        
        
        def update(val):
            ax2.cla()
            ax1.imshow(self.AsinhStretch(black_slider.val, stretch_slider.val), vmin=0, vmax=1, cmap='gray')
            ax2.hist(self.AsinhStretch(black_slider.val, stretch_slider.val).ravel(), 256, (0, 1))
            fig.canvas.draw_idle()
            
        stretch_slider.on_changed(update)
        black_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to apply changes or reset the sliders to initial values.
        resetax = fig.add_axes([0.5, 0., 0.1, 0.05])
        button_reset = Button(resetax, 'Reset', hovercolor='0.5')

        applyax = fig.add_axes([0.3, 0., 0.1, 0.05])
        button_apply = Button(applyax, 'Apply', hovercolor='0.5')

        def reset(event):
            stretch_slider.reset()
            black_slider.reset()
            
        button_reset.on_clicked(reset)

        def apply(event):
            self.image = self.AsinhStretch(black_slider.val, stretch_slider.val)
            stretch_slider.reset()
            black_slider.reset()

        button_apply.on_clicked(apply)

        plt.show()
        return self.image
    

class Mtf():
    def __init__(self, image):
        self.image = image
    
    def mtf(self, midtones, shadows, highlights):
        xp = (self.image - shadows) / (highlights - shadows)
        return ((midtones - 1) * xp) / ((2 * midtones - 1) * xp - midtones)
    
    def plot_mtf(self, m=0.5, s=0, h=1):
        fig, (ax1, ax2) = plt.subplots(2, height_ratios=[2, 0.5])
        ax1.imshow(self.image, vmin=0, vmax=1, cmap='gray')
        ax2.hist(self.image.ravel(), 256, (0, 1))

        fig.subplots_adjust(bottom=0.25)

        axmidtones = fig.add_axes([0.2, 0.15, 0.6, 0.02])
        midtones_slider = Slider(
            ax=axmidtones,
            label="midtones ",
            valmin=0,
            valmax=1,
            valinit=m
        )

        axshadows = fig.add_axes([0.2, 0.1, 0.6, 0.02])
        shadows_slider = Slider(
            ax=axshadows,
            label="shadows",
            valmin=0,
            valmax=1,
            valinit=s
        )
        
        axhighlights = fig.add_axes([0.2, 0.05, 0.6, 0.02])
        highlights_slider = Slider(
            ax=axhighlights,
            label="highlights",
            valmin=0,
            valmax=1,
            valinit=h
        )
        
        def update(val):
            ax2.cla()
            ax1.imshow(self.mtf(midtones_slider.val, shadows_slider.val, highlights_slider.val), vmin=0, vmax=1, cmap='gray')
            ax2.hist(self.mtf(midtones_slider.val, shadows_slider.val, highlights_slider.val).ravel(), 256, (0, 1))
            fig.canvas.draw_idle()
            
        midtones_slider.on_changed(update)
        shadows_slider.on_changed(update)
        highlights_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to apply changes or reset the sliders to initial values.
        resetax = fig.add_axes([0.5, 0., 0.1, 0.05])
        button_reset = Button(resetax, 'Reset', hovercolor='0.5')

        applyax = fig.add_axes([0.3, 0., 0.1, 0.05])
        button_apply = Button(applyax, 'Apply', hovercolor='0.5')

        def reset(event):
            midtones_slider.reset()
            shadows_slider.reset()
            highlights_slider.reset()
            
        button_reset.on_clicked(reset)

        def apply(event):
            self.image = self.mtf(midtones_slider.val, shadows_slider.val, highlights_slider.val)
            midtones_slider.reset()
            shadows_slider.reset()
            highlights_slider.reset()

        button_apply.on_clicked(apply)
        
        plt.show()
        return self.image

anh = Stretch(img)
imag = anh.plot_asinh()

mtf = Mtf(img)
mt = mtf.plot_mtf()

