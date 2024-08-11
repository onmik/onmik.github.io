import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
from PIL import Image
import arithm

img = np.asarray(Image.open('linear.tif'))

class GHS:
    def __init__(self, image):
        self.image = image
        
    def ghs(self, stretch, D, b, SP, LP, HP):
        
        # ----------------- normal ghs --------------------------
        if stretch == 1: 
            if b == -1:                       
                qlp = -1 * np.log1p(D * (SP - LP))
                q0 = qlp - D * LP / (1 + D * (SP - LP))
                qwp = np.log1p(D * (HP - SP))
                q1 = qwp + D * (1 - HP) / (1 + D * (HP - SP))
                q = 1 / (q1 - q0)
                
                # coefficients for img < LP
                b1 = (1 + D * (SP - LP)) / (D * q)
                    
                # coefficients for img < SP
                a2 = (-q0) * q
                b2 = -q
                c2 = 1 + D * SP
                d2 = -D
                
                # coefficints for SP <= img <=HP
                a3 = (-q) * q
                b3 = q
                c3 = 1 - D * SP
                d3 = D
                
                # coefficients for img > HP
                a4 = (qwp - q0 - D * HP / (1 + D * (HP - SP))) * q
                b4 = q * D / (1 + D * (HP - SP))
            
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
                a1 = 0
                b1 = D * (np.sign(1 - D * b * (SP - LP)) * np.power(np.abs(1 - D * b * (SP - LP)),
                                       1 / b)) * q
                
                # coefficients for LP <= img < SP
                a2 = -(1 / (b + 1) + q0) * q
                b2 = q / (b + 1)
                c2 = 1 - D * b * SP
                d2 = D * b
                e2 = (b + 1) / b
                
                # coefficints for SP <= img <=HP
                a3 = (1 / (b + 1) - q0) * q
                b3 = -q / (b + 1)
                c3 = 1 + D * b * SP
                d3 = -D * b
                e3 = (b + 1.0) / b
                
                # coefficients for img > HP
                a4 = (qwp - q0 - D * HP * (np.sign(1 - D*b*(HP - SP)) * np.power(np.abs(1 - D*b*(HP - SP)), 
                                                        1 / b))) * q
                b4 = D * (np.sign(1 - D * b * (HP - SP)) * np.power(np.abs(1 - D * b * (HP - SP)),1 / b)) * q
                
            # Exponential GHS
            elif(b ==0):
                qlp = np.exp(-D * (SP - LP))
                q0 = qlp - D * LP * qlp
                qwp = 2 - np.exp(-D * (HP - SP))
                q1 = qwp + D * (1 - HP) * (2 - qwp)
                q = 1 / (q1 - q0)
                
                # coefficients for img < LP
                a1 = 0
                b1 = D * qlp * q
                
                # coefficients for LP <= img < SP
                a2 = -q0 * q
                b2 = q
                c2 = -D * SP
                d2 = D
                e2 = 0
                
                # coefficients for SP<=img<=HP
                a3 = (2 - q0) * q
                b3 = -q
                c3 = D * SP
                d3 = -D
                e3 = 0
                
                # coefficients for img > HP
                a4 = (qwp - q0 - D * HP * (2 - qwp)) * q
                b4 = D * (2 - qwp) * q
                
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
                a1 = 0
                b1 = D * (np.sign(1 + D * b * (SP - LP)) * np.power(np.abs(1 + D * b * (SP - LP)), 
                                       -(1 + b) / b)) * q
                
                # coefficients for LP<=img<SP
                a2 = -q0 * q
                b2 = q
                c2 = 1 + D * b * SP
                d2 = -D * b
                e2 = -1/b
                
                # coefficients for SP <= img <=HP
                a3 = (2 - q0) * q
                b3 = -q
                c3 = 1 - D * b * SP
                d3 = D * b
                e3 = -1 / b
                
                # coessicients for img > HP
                a4 = (qwp-q0-D * HP * (np.sign(1 + D * b * (HP - SP)) * np.power(np.abs(1 + D * b * (HP - SP)), 
                                                    -(b + 1) / b))) * q
                b4 = (D * (np.sign(1 + D * b * (HP - SP)) * np.power(np.abs(1 + D * b * (HP - SP)), 
                                        -(b + 1) / b))) * q
            
            
            if D ==1e-10:
                out = self.image
              
            if b == -1:             
                res1 = a2 + b2 * np.log(c2 + d2 * self.image)
                res2 = a3 + b3 * np.log(c3 + d3 * self.image)
            elif b < 0 or b > 0:
                res1 = a2 + b2 * (np.sign(c2 + d2 * self.image) * np.power(np.abs(c2 + d2 * self.image), e2))
                res2 = a3 + b3 * (np.sign(c3 + d3 * self.image) * np.power(np.abs(c3 + d3 * self.image), e3))
            else:
                res1 = a2 + b2 * np.exp(c2 + d2 * self.image)
                res2 = a3 + b3 * np.exp(c3 + d3 * self.image)
                          
            out =  np.where(self.image < LP, b1 * self.image, 
                            np.where(self.image < SP, res1, 
                                     np.where(self.image < HP, res2,
                                             a4 + b4 * self.image)))
        
        #------------------------ inverse ghs --------------------------------
        elif stretch == 2:
            if b == -1:
                qlp = -1 * np.log1p(D * (SP - LP))
                q0 = qlp - D * LP / (1 + D * (SP - LP))
                qwp = np.log1p(D * (HP - SP))
                q1 = qwp + D * (1 - HP) / (1 + D * (HP - SP))
                q = 1 / (q1 - q0)
                
                LPT = (qlp - q0) * q
                SPT = q0 * q
                HPT = (qwp - q0) * q
                
                # coefficients for img < LP
                b1 = (1 + D * (SP - LP)) / (D * q)
                
                a2 = (1 + D * SP) / D
                b2 = -1 / D
                c2 = -q0
                d2 = -1 / q
                
                a3 = -(1 - D * SP) / D
                b3 = 1 / D
                c3 = q0
                d3 = 1 / q
                
                a4 = HP + (q0 - qwp) * (1 + D * (HP - SP)) / D
                b4 = (1 + D * (HP - SP)) / (q * D)
                
            elif b < 0:
                b = -b
                qlp = (1 - arithm.pow((1 + D * b * (SP - LP)), (b - 1) / b)) / (b - 1)
                q0 = qlp - D * LP * (arithm.pow((1 + D * b * (SP - LP)), -1 / b))
                qwp = (arithm.pow((1 + D * b * (HP - SP)), (b - 1) / b) - 1) / (b - 1)
                q1 = qwp + D * (1 - HP) * (arithm.pow((1 + D * b * (HP - SP)), -1 / b))
                q = 1 / (q1 - q0)
                LPT = (qlp - q0) * q
                SPT = q0 * q
                HPT = (qwp - q0) * q
                
                b1 = arithm.pow(1 + D * b * (SP - LP), 1 / b) / (q * D)
                
                a2 = (1 + D * b * SP) / (D * b)
                b2 = -1 / (D * b)
                c2 = -q0 * (b - 1) + 1
                d2 = (1 - b) / q
                e2 = b / (b - 1)

                a3 = (D * b * SP - 1) / (D * b)
                b3 = 1 / (D * b)
                c3 = 1 + q0 * (b - 1)
                d3 = (b - 1) / q
                e3 = b / (b - 1)
                
                a4 = (q0 - qwp) / (D * arithm.pow((1 + D * b * (HP - SP)), -1 / b)) + HP
                b4 = 1 / (D * arithm.pow((1 + D * b * (HP - SP)), -1 / b) * q)
                
            elif b == 0:
                qlp = np.exp(-D * (SP - LP))
                q0 = qlp - D * LP * np.exp(-D*(SP - LP))
                qwp = 2 - np.exp(-D * (HP -SP))
                q1 = qwp + D * (1 - HP) * np.exp(-D * (HP - SP))
                q = 1 / (q1 - q0)
                    
                LPT = (qlp-q0)*q
                SPT = (1-q0)*q
                HPT = (qwp-q0)*q
                    
                a1 = 0
                b1 = 1/(D * np.exp(-D * (SP - LP)) * q)
                
                a2 = SP
                b2 = 1 / D
                c2 = q0
                d2 = 1/q
                
                a3 = SP
                b3 = -1 / D
                c3 = (2 - q0)
                d3 = -1 / q
                
                a4 = (q0 - qwp)/(D * np.exp(-D * (HP - SP))) + HP
                b4 = 1/(D * np.exp(-D * (HP - SP)) * q)
                
            else: # b > 0
                qlp = arithm.pow(( 1 + D * b * (SP - LP)), -1/b)
                q0 = qlp - D * LP * arithm.pow((1 + D * b * (SP - LP)), -(1 + b) / b)
                qwp = 2 - arithm.pow(1 + D * b * (HP - SP), -1 / b)
                q1 = qwp + D * (1 - HP) * arithm.pow((1 + D * b * (HP - SP)), -(1 + b) / b)
                q = 1 / (q1 - q0)
                
                LPT = (qlp-q0)*q
                SPT = (1-q0)*q
                HPT = (qwp-q0)*q
                
                b1 = 1/(D * arithm.pow((1 + D * b * (SP - LP)), -(1 + b) / b) * q)
                
                a2 = 1 / (D * b) + SP
                b2 = -1/(D * b)
                c2 = q0
                d2 = 1 / q
                e2 = -b
                
                a3 = -1 / (D * b) + SP
                b3 = 1 / (D * b)
                c3 = (2 - q0)
                d3 = -1 / q
                e3 = -b
                
                a4 = (q0-qwp)/(D * arithm.pow((1 + D * b * (HP - SP)), -(b + 1) / b))+HP
                b4 = 1/((D * arithm.pow((1 + D * b * (HP - SP)), -(b + 1) / b)) * q)

            if D == 0:
                out = self.image
            elif b == -1:
                res1 = a2 + b2 * np.exp(c2 + d2 * self.image)
                res2 = a3 + b3 * np.exp(c3 + d3 * self.image)
            elif b < 0 or b > 0:
                res1 = a2 + b2 * arithm.pow(c2 + d2 * self.image, e2)
                res2 = a3 + b3 * arithm.pow(c3 + d3 * self.image, e3)
            else:
                res1 = a2 + b2 * np.log(c2 + d2 * self.image)
                res2 = a3 + b3 * np.log(c3 + d3 * self.image)  

            out = np.where(self.image < LPT, b1 * self.image,
                           np.where(self.image < SPT, res1,
                                    np.where(self.image < HPT, res2,
                                             a4 + b4 * self.image)))
        # ----------------------- modified asinh ----------------------------  
        elif stretch == 3:
            qlp = -np.log(D * (SP - LP) + arithm.pow((D * D * (SP - LP) * (SP - LP) + 1), 0.5))
            q0 = qlp - LP * D * arithm.pow((D * D * (SP - LP) * (SP - LP) + 1), -0.5)
            qwp = np.log(D * (HP - SP) + arithm.pow((D * D * (HP - SP) * (HP - SP) + 1), 0.5))
            q1 = qwp + (1 - HP) * D * arithm.pow((D * D * (HP - SP) * (HP - SP) +1), -0.5)
            q = 1 / (q1 - q0)
            
            a1 = 0
            b1 = D * arithm.pow((D * D * (SP - LP) * (SP - LP) + 1),-0.5)*q
            
            a2 = -q0 * q
            b2 = -q
            c2 = -D
            d2 = D * D
            e2 = SP
            
            a3 = -q0 * q
            b3 = q
            c3 = D
            d3 = D * D
            
            e3 = SP
            a4 = (qwp - HP * D * arithm.pow((D * D * (HP - SP) * (HP - SP) + 1), -0.5) - q0) * q
            b4 = D * arithm.pow((D * D * (HP - SP) * (HP - SP) + 1), -0.5) * q

        
        return out
                
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











            