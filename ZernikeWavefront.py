import numpy as np
from matplotlib import pyplot as plt
from zernike import RZern

class ZernikeWF():
    def __init__(self):
        self.N = 256
        self.L = 10   # length of one side of the computational domain in mm
        self.L_g = 0.256 #ganglioin field of view in mm    
        self.x = np.linspace(-self.L/2, self.L/2, self.N)
        self.y = np.linspace(-self.L/2, self.L/2, self.N)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.Zer_div = np.sqrt(2)
        self.Zernike_Aberrations = ["Piston", "X-Tilt", "Y-Tilt", "Defocus", "Oblique Astigmatism",
                                "Vertical Astigmatism", "Vertical Coma", "Horizontal Coma", 
                                "Vertical Trefoil", "Oblique Trefoil", "Primary Spherical", "Vertical Secondary Astigmatism",
                               "Oblique Secondary Astigmatism", "Vertical Quadrafoil", "Oblique Quadrafoil"]
        self.Cmap = ["binary", "bone", "seismic", "gray", "Greys", "Greys_r", "purples", "Purples_r", "Spectral"]
        
    def lens_FT(self, U_in):
        
        U_in = np.fft.fftshift(U_in) # shift U_in in order to perform the fft correct
        U_out = np.fft.fftshift(np.fft.fft2(U_in))
        return U_out

    def lens_inv_FT(self, U_in):
        U_in = np.fft.ifftshift(U_in)
        U_out = np.fft.ifftshift(np.fft.ifft2(U_in))
        return U_out
      
    def Zernike_Wavefront_Generator(self, Zernike_name, coefficient = 1):
        # Generate Zernike Polynomial-determined Wavefront Aberrations
        # Zernike_name: An array containing n Zernike polynomials names in order
        
        cart = RZern(4)
        cart.make_cart_grid(self.xx/(self.L/self.Zer_div), self.yy/(self.L/self.Zer_div))
        Zer_num = cart.nk
        c = np.zeros(cart.nk)
        for i in range(0,Zer_num):
            if self.Zernike_Aberrations[i] == Zernike_name:
                c[i] = 1
        
        Wavefront = cart.eval_grid(c, matrix=True) #Zernike Polynomials
        Wavefront = np.exp(coefficient*1j*Wavefront) #Wavefront
        
        return Wavefront
     
    def Multi_ZernikeWF_Gen(self, co):
        # Generate multiple Zernike polynomials represented wavefront with coefficient
        # co: 1d array of numbers containing Zernike mode coefficients in order
        Wavefront = np.ones((self.N, self.N))
        for i in range(1,15):
            Zer_n = i
            nn = self.Zernike_Aberrations[Zer_n - 1]
            sub_wavefront = self.Zernike_Wavefront_Gen(nn, co[i]) #get zernike wavefront
            Temp = np.multiply(sub_wavefront, Wavefront)
            Wavefront = Temp
         
        return Wavefront
      
    def Image_gen(self, Original_Image, Wavefront):
        # Generate Final Image of 4f system
        Image_F = self.lens_FT(Original_Image)
        
        Wavefront_invF = self.lens_inv_FT(Wavefront)  # PSI
        PSF = (np.real(Wavefront_invF))**2    # Point Spread Function
        
        # Multiply Image_F and inverse FT of PSF to prevent convolution
        PSF_F = self.lens_FT(PSF)  
        F_all = np.multiply(Image_F, PSF_F)
        Final_Image_withphase = self.lens_inv_FT(F_all)
        Final_Image = np.sqrt((np.real(Final_Image_withphase)**2))
        
        return Final_Image
      
    def metric_function(self, I):
        # Define metric function of images: square root of intensity squared and summed
        # I: Image
        metric = 0
        temp = 0
        for i in range(self.N):
            for j in range(self.N):
                temp = I[i][j]**2
                metric = metric + temp
                temp = 0
                
        metric = np.sqrt(metric)
        return metric
      
    def Image_Recovery(self, Image, WF):
        # Image recovery process with known wavefront pattern
        Image_F = self.lens_FT(Image)
        WF_invF = self.lens_inv_FT(WF)
        PSF = np.real(WF_invF)**2
        PSF_F = self.lens_FT(PSF)
        F_recover = np.multiply(Image_F * PSF_F**(-1))
        I_recover = self.lens_inv_FT(F_recover) 
        I_recover_real = np.abs(np.real(I_recover))
        
        return I_recover_real

    def Get_Origin_Img(self):
        # get original .tif image
        I = plt.imread('abganglion.tif')
        Intensity = np.zeros((256,256))
        for i in range(0,255):
            for j in range(0,255):
                Intensity[i][j] = I[i][j][2]/255
        
        return Intensity
    
    def Graphs_All(self, Origin, Final, WF, Zer, NO = '0'):
        # Plot original image, final image, and wavefront pattern in one figure
        WF = np.real(WF)    
        fig1, (ax1, ax2, ax3) = plt.subplots(figsize = (18,5), ncols = 3)
        
        org  = ax1.imshow(Origin, extent =[-self.L_g/2,self.L_g/2,-self.L_g/2,self.L_g/2], cmap =self.Cmap[1])
        fig1.colorbar(org, ax=ax1)
        ax1.set_xlabel("x-FOV/mm")
        ax1.set_ylabel("y-FOV/mm")
        ax1.title.set_text("Ganglion")
        
        final = ax2.imshow(Final, extent =[-self.L_g/2,self.L_g/2,-self.L_g/2,self.L_g/2], cmap =self.Cmap[1])
        fig1.colorbar(final, ax=ax2)
        ax2.set_xlabel("x-FOV/mm")
        ax2.set_ylabel("y-FOV/mm")
        ax2.title.set_text("Final Image with " + Zer)
        
        WF1 = np.real(WF)
        wf = ax3.imshow(WF1, extent =[-self.L/2,self.L/2,-self.L/2,self.L/2], cmap = "viridis")
        fig1.colorbar(wf, ax=ax3)
        ax3.set_xlabel("${L_x}$/mm")
        ax3.set_ylabel("${L_y}$/mm")
        ax3.title.set_text(Zer + " Wavefront")
        
        #plt.savefig(NO + '. ' + Zer + '.jpeg', dpi = 300)
        plt.show()
        
    def generate_Img(self, I, co = [[0,0,0,0,0]]):
        # generate image with wavefront of selected 5 modes. Coefficients are from DRL actions.
        # I: Image to be changed 
        # co: 5-element list of coefficients for 5 modes
        Co = np.zeros(15)
        Co[3] = co[0][0]
        Co[4] = co[0][1]
        Co[5] = co[0][2]
        Co[7] = co[0][3]
        Co[8] = co[0][4]
        WF = self.Multi_ZernikeWF_Gen(Co)
        Img = self.Image_Recovery(I, WF)
        return Img
  
  
  
  
  
  
  
  
