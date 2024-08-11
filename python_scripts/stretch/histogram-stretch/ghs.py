import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
from PIL import Image

img = np.asarray(Image.open('linear.tif'))

class GHS:
    def __init__(self, image):
        
        self.image = image
        
        self.a1 = 0
        self.b1 = 0
        
        self.a2 = 0
        self.b2 = 0
        self.c2 = 0
        self.d2 = 0
        self.e2 = 0
        
        self.a3 = 0
        self.b3 = 0
        self.c3 = 0
        self.d3 = 0
        self.e3 = 0
        
        self.a4 = 0
        self.b4 = 0
        
    def coeffs(self, D, b, SP, LP, HP):
        # Logarithmic GHS
        if b == -1:                       
            qlp = -1 * np.log(1 + D * (SP - LP))
            q0 = qlp - D * LP / (1 + D * (SP - LP))
            qwp = np.log(1 + D * (HP - SP))
            q1 = qwp + D * (1 - HP) / (1 + D * (HP - SP))
            q = 1 / (q1 - q0)
            
            # coefficients for img < LP
            self.a1 = 0
            self.b1 = D / (1 + D * (SP - LP)) * q
                
            # coefficients for img < SP
            self.a2 = (-q0) * q
            self.b2 = -q
            self.c2 = 1 + D * SP
            self.d2 = -D
            self.e2 = 0
            
            # coefficints for SP <= img <=HP
            self.a3 = (-q) * q
            self.b3 = q
            self.c3 = 1 - D * SP
            self.d3 = D
            self.e3 = 0
            
            # coefficients for img > HP
            self.a4 = (qwp - q0 - D * HP / (1 + D * (HP - SP))) * q
            self.b4 = q * D / (1 + D * (HP - SP))
        
            # Integral GHS    
        elif (b < 0):
            qlp = -(1 - np.sign(1 - D * b * (SP - LP)) * np.power(np.abs(1 - D * b * (SP - LP)), 
                                 (b + 1) / b)) / (b + 1)
            q0 = qlp - D * LP * (np.sign(1 - D * b * (SP - LP)) * np.power(np.abs(1 - D * b * (SP - LP)), 1 / b))
            qwp = -(np.sign(1 - D * b * (HP - SP)) * np.power(np.abs(1 - D * b * (HP - SP)), 
                             (b + 1) / b) - 1) / (b + 1)
            q1 = qwp + D * (1 - HP) * (np.sign(1 - D * b * (HP - SP)) * np.power(np.abs(1 - D * b * (HP - SP)),
                                               1 / b))
            q = 1 / (q1 - q0)
            
            # coefficients for img < LP
            self.a1 = 0
            self.b1 = D * (np.sign(1 - D * b * (SP - LP)) * np.power(np.abs(1 - D * b * (SP - LP)),
                                   1 / b)) * q
            
            # coefficients for LP <= img < SP
            self.a2 = -(1 / (b + 1) + q0) * q
            self.b2 = q / (b + 1)
            self.c2 = 1 - D * b * SP
            self.d2 = D * b
            self.e2 = (b + 1) / b
            
            # coefficints for SP <= img <=HP
            self.a3 = (1 / (b + 1) - q0) * q
            self.b3 = -q / (b + 1)
            self.c3 = 1 + D * b * SP
            self.d3 = -D * b
            self.e3 = (b + 1.0) / b
            
            # coefficients for img > HP
            self.a4 = (qwp - q0 - D * HP * (np.sign(1 - D*b*(HP - SP)) * np.power(np.abs(1 - D*b*(HP - SP)), 
                                                    1 / b))) * q
            self.b4 = D * (np.sign(1 - D * b * (HP - SP)) * np.power(np.abs(1 - D * b * (HP - SP)),1 / b)) * q
            
        # Exponential GHS
        elif(b ==0):
            qlp = np.exp(-D * (SP - LP))
            q0 = qlp - D * LP * qlp
            qwp = 2 - np.exp(-D * (HP - SP))
            q1 = qwp + D * (1 - HP) * (2 - qwp)
            q = 1 / (q1 - q0)
            
            # coefficients for img < LP
            self.a1 = 0
            self.b1 = D * qlp * q
            
            # coefficients for LP <= img < SP
            self.a2 = -q0 * q
            self.b2 = q
            self.c2 = -D * SP
            self.d2 = D
            self.e2 = 0
            
            # coefficients for SP<=img<=HP
            self.a3 = (2 - q0) * q
            self.b3 = -q
            self.c3 = D * SP
            self.d3 = -D
            self.e3 = 0
            
            # coefficients for img > HP
            self.a4 = (qwp - q0 - D * HP * (2 - qwp)) * q
            self.b4 = D * (2 - qwp) * q
            
        # Hyperbolic/Harmonic GHS
        else: # (b > 0)
            qlp = np.sign(1 + D * b * (SP - LP)) * np.power(np.abs(1 + D * b * (SP - LP)), -1 / b)
            q0 = qlp - D * LP * (np.sign(1 + D * b * (SP - LP)) * np.power(np.abs(1 + D * b * (SP - LP)),
                                         -(1.0 + b) / b))
            qwp = 2 - np.sign(1 + D * b * (HP - SP)) * np.power(np.abs(1 + D * b * (HP - SP)), -1 / b)
            q1 = qwp + D * (1 - HP) * (np.abs(1 + D * b * (HP - SP)) * np.power(np.abs(1 + D * b * (HP - SP)),
                                               -(1 + b) / b))
            q = 1 / (q1 - q0)
            
            # coefficients for img < LP
            self.a1 = 0
            self.b1 = D * (np.sign(1 + D * b * (SP - LP)) * np.power(np.abs(1 + D * b * (SP - LP)), 
                                   -(1 + b) / b)) * q
            
            # coefficients for LP<=img<SP
            self.a2 = -q0 * q
            self.b2 = q
            self.c2 = 1 + D * b * SP
            self.d2 = -D * b
            self.e2 = -1/b
            
            # coefficients for SP <= img <=HP
            self.a3 = (2 - q0) * q
            self.b3 = -q
            self.c3 = 1 - D * b * SP
            self.d3 = D * b
            self.e3 = -1 / b
            
            # coessicients for img > HP
            self.a4 = (qwp-q0-D * HP * (np.sign(1 + D * b * (HP - SP)) * np.power(np.abs(1 + D * b * (HP - SP)), 
                                                -(b + 1) / b))) * q
            self.b4 = (D * (np.sign(1 + D * b * (HP - SP)) * np.power(np.abs(1 + D * b * (HP - SP)), 
                                    -(b + 1) / b))) * q
            
            
        
    
    def ghs(self, D, b, SP, LP, HP):
        self.coeffs(D, b, SP, LP, HP)
        
        if D ==1e-10:
            return self.image
        
        if b == -1:             
            res1 = self.a2 + self.b2 * np.log(self.c2 + self.d2 * self.image)
            res2 = self.a3 + self.b3 * np.log(self.c3 + self.d3 * self.image)
        elif b < 0 or b > 0:
            res1 = self.a2 + self.b2 * (np.sign(self.c2 + self.d2 * self.image) * np.power(np.abs(self.c2 + self.d2 * self.image), self.e2))
            res2 = self.a3 + self.b3 * (np.sign(self.c3 + self.d3 * self.image) * np.power(np.abs(self.c3 + self.d3 * self.image), self.e3))
        else:
            res1 = self.a2 + self.b2 * np.exp(self.c2 + self.d2 * self.image)
            res2 = self.a3 + self.b3 * np.exp(self.c3 + self.d3 * self.image)
                    
        return np.where(self.image < LP, self.b1 * self.image, 
                        np.where(self.image < SP, res1, 
                                 np.where(self.image < HP, res2,
                                          self.a4 + self.b4 * self.image)))
                                            #return out
                
                
    def plot(self, D=0, b=0, SP=0, LP=0, HP=1):
        fig, ax = plt.subplots()
        ax.imshow(self.image, vmin=0, vmax=1, cmap='gray')
        
        fig.subplots_adjust(bottom=0.4)
        
        axD = fig.add_axes([0.2, 0.3, 0.6, 0.02])
        D_slider = Slider(
            ax=axD,
            label="D ",
            valmin=1e-10,
            valmax=50,
            valinit=D
            )

        axb = fig.add_axes([0.2, 0.25, 0.6, 0.02])
        b_slider = Slider(
            ax=axb,
            label="b ",
            valmin=-5,
            valmax=15,
            valinit=b
            )
             
        axSP = fig.add_axes([0.2, 0.2, 0.6, 0.02])
        SP_slider = Slider(
            ax=axSP,
            label="SP ",
            valmin=0,
            valmax=1,
            valinit=SP
            )
             
        axLP = fig.add_axes([0.2, 0.15, 0.6, 0.02])
        LP_slider = Slider(
            ax=axLP,
            label="LP ",
            valmin=0,
            valmax=1,
            valinit=LP
            )
             
        axHP = fig.add_axes([0.2, 0.1, 0.6, 0.02])
        HP_slider = Slider(
            ax=axHP,
            label="HP ",
            valmin=0,
            valmax=1,
            valinit=HP
            )
             
        def update(val):
            #self.coeffs(D_slider.val, b_slider.val, SP_slider.val, LP_slider.val, HP_slider.val)
            ax.imshow(self.ghs(D_slider.val, b_slider.val, SP_slider.val, LP_slider.val, HP_slider.val), 
                      vmin=0, vmax=1, cmap='gray')
            fig.canvas.draw_idle()
            
        D_slider.on_changed(update)
        b_slider.on_changed(update)
        SP_slider.on_changed(update)
        LP_slider.on_changed(update)
        HP_slider.on_changed(update)
            
        # Create a `matplotlib.widgets.Button` to apply changes or reset the sliders to initial values.
        resetax = fig.add_axes([0.5, 0., 0.1, 0.05])
        button_reset = Button(resetax, 'Reset', hovercolor='0.5')

        applyax = fig.add_axes([0.3, 0., 0.1, 0.05])
        button_apply = Button(applyax, 'Apply', hovercolor='0.5')

        def reset(event):
            D_slider.reset()
            b_slider.reset()
            SP_slider.reset()
            LP_slider.reset()
            HP_slider.reset()
            
        button_reset.on_clicked(reset)

        def apply(event):
            #self.coeffs(D_slider.val, b_slider.val, SP_slider.val, LP_slider.val, HP_slider.val)
            self.image = self.ghs(D_slider.val, b_slider.val, SP_slider.val, LP_slider.val, HP_slider.val)
            
            D_slider.reset()
            b_slider.reset()
            SP_slider.reset()
            LP_slider.reset()
            HP_slider.reset()
            
        button_apply.on_clicked(apply)
        
        plt.show()
        

stretch = GHS(img)
imag = stretch.plot()

#stretch.coeffs(50, 1, 0, 0, 1)
Ghs = stretch.ghs(50, 10, 0, 0, 1)
plt.imshow(Ghs)











            