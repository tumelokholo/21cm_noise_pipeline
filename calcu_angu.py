import numpy as np
import pylab as pl
import matplotlib.colors as colors
from scipy.interpolate import interp1d

#ndimension
ndim = 140

#reading a 21cm box
box = np.fromfile("deltaTb_z8.000_fesc_0.45_A_5.99e+39_C_0.22.dat", dtype=np.float32)

#subtracting the mean from the box
box = box - np.mean(box)

#reshaping the box
box = box.reshape(ndim,ndim,ndim)*1e3

#Fourier tranforming the box
FFT = np.fft.fftn(box,norm="ortho")
#print FFT.shape

#Creating frequency bin for the Given a window length n=140 and a sample spacing d=0.535:
fftfreq = 2*np.pi*np.fft.fftfreq(140,0.535)
#print fftfreq.shape

#Calculating the k_max
b_max = 876  #ska=5834, hera=876, lofar=3696 maximum baseline of instrument
lambda1 = 21e-2  #wavelength
z = 8   #redshift
D = 8939.6  # Comoving distance

k_max = (2*np.pi*b_max)/((1+z)*lambda1*D)
print(k_max)

##The loop to account for Angular resolution and Foreground cleaning.
m = 1.0
for i in range(ndim):
    for j in range(ndim):
        for l in range(ndim):
            k_per = np.sqrt(fftfreq[i]**2.0 + fftfreq[j]**2.0)
            if k_per > k_max:
                FFT[i,j,l] = 0
                #continue
            k_par = fftfreq[l]
            if k_per*m > k_par:
                FFT[i,j,l] = 0

#Inverse Fourier of the box
IFFT2 = np.fft.ifftn(FFT,norm="ortho")

#Taking the real part of the box
rFFT = np.real(IFFT2)

los = 75

##Generating Thermal Noise

#Creating a White noise box
print "starting the white noise calculation!"
white_noise = np.array([[[np.random.normal(0.0, 1.0) for col in range(140)] for row in range(140)] for x in range(140)])

#
k = 2*np.pi*np.fft.fftfreq(140,0.535)


#
data  = np.load('all_noise.npz')

ks_s, Ts_s = data["ska"]
p_s     = Ts_s*2.0*np.pi**2.0/ks_s**3.0
f_s     = interp1d(np.log10(ks_s), np.log10(p_s))

ks_h, Ts_h = data["hera"]
p_h     = Ts_h*2.0*np.pi**2.0/ks_h**3.0
f_h     = interp1d(np.log10(ks_h), np.log10(p_h))

ks_l, Ts_l = data["lofar"]
p_l     = Ts_l*2.0*np.pi**2.0/ks_l**3.0
f_l     = interp1d(np.log10(ks_l), np.log10(p_l))
'''
pl.figure(1)
pl.loglog(ks_s,p_s,label="ska")
pl.loglog(ks_h,p_h,label="hera")
pl.loglog(ks_l,p_l,label="lofar")
pl.xlabel("k/[Mpc]")
pl.ylabel("P(k)")
pl.legend()
#pl.show()
'''
#Fourier trasforming the white noise
FT_s   = np.fft.fftn(white_noise,  norm="ortho")
FT_h   = np.fft.fftn(white_noise,  norm="ortho")
FT_l   = np.fft.fftn(white_noise,  norm="ortho")

print "starting loop!"

for m in range(140):
    for n in range(140):
        for o in range(140):
            kbar = np.sqrt(k[m]**2.0 + k[n]**2.0 + k[o]**2.0)
            if min(ks_l) <=  kbar  <= max(ks_l):
                FT_s[m,n,o]   *= np.sqrt(10.0**f_s(np.log10(kbar)))
                FT_h[m,n,o]   *= np.sqrt(10.0**f_h(np.log10(kbar)))
                FT_l[m,m,o]   *= np.sqrt(10.0**f_l(np.log10(kbar)))

IFT_s        = np.fft.ifftn(FT_s, norm="ortho").real/(0.535**(3.0/2.0))
IFT_h        = np.fft.ifftn(FT_h, norm="ortho").real/(0.535**(3.0/2.0))
IFT_l        = np.fft.ifftn(FT_l, norm="ortho").real/(0.535**(3.0/2.0))



rFT_box_s    = rFFT + IFT_s
rFT_box_h    = rFFT + IFT_h
rFT_box_l    = rFFT + IFT_l

print "starting the plot!"
'''
v_max = np.max([np.max(rFT_box_s[:,:,los]),np.max(box[:,:,los])])
v_min = np.min([np.min(rFT_box_s[:,:,los]),np.min(box[:,:,los])])
v_ma = np.max([np.max(rFT_box_l[:,:,los]),np.max(box[:,:,los])])
v_mi = np.min([np.min(rFT_box_l[:,:,los]),np.min(box[:,:,los])])
color_map = "coolwarm"
pl.figure(3)
pl.subplot(131)
pl.imshow(box[:,:,los], vmin=v_min, vmax=v_max,  norm=colors.SymLogNorm(linthresh=10**-3))
pl.title("Original box")
pl.set_cmap(color_map)
#pl.savefig("origim"+n[i]+".png")

#pl.figure(2)
pl.subplot(132)
pl.imshow(rFFT[:,:,los],vmin=v_min, vmax=v_max,  norm=colors.SymLogNorm(linthresh=10**-3))
pl.title("AR and FG cleaned box(Lofar)")
pl.set_cmap(color_map)
#pl.savefig("origim"+n[i]+".png")

pl.subplot(133)
pl.imshow(rFT_box_l[:,:,los],vmin=v_mi, vmax=v_ma,  norm=colors.SymLogNorm(linthresh=10**-3) )
pl.title("TN + cleaned box(Lofar)")
pl.set_cmap(color_map)
#pl.savefig("im.png")
pl.show()

'''

##Fuction to extract the power spectrum from the box
print "starting the function of extraction of PS!"
box1 = rFT_box_s
N = 140    #number of cells
dl = 0.535 #cell size



def PowerSpectrum(box1,N,dl):
    FT_box  = np.fft.fftn(box1,norm="ortho") # norm is important to be ortho, otherwise normalization is wrong!
    k1       = 2*np.pi*np.fft.fftfreq(N,dl)
    pk      = np.zeros(N)   
    count   = np.zeros(N)    # to count powers falling in each bin
    dk      = 2*np.pi/(N*dl) # dk is the bin size
    for a in range(N):
        for b in range(N):
            for c in range(N):
                kbar1      = np.sqrt(k1[a]**2.0 + k1[b]**2.0 + k1[c]**2.0) # get the k_bar in 3D
                t         = int(round(kbar1/dk)) # find the corresponding bin number 
            #t = kbar1/dk
            #t //= round(t)
            #print t
                count[t] += 1.0
                pk[t]    += FT_box[a,b,c]*np.conj(FT_box[a,b,c])
    pk       /= count # average each bin
    pk       *= (dl)**3.0 
    dk        = np.arange(float(N))*dk

    return  dk, pk

d = PowerSpectrum(box1,140,0.535)

pl.figure(5)
pl.loglog(d[0],d[1])
pl.loglog(ks_h,p_h)
pl.xlabel("k/[Mpc]")
pl.ylabel("P(k)")
pl.title("ska")
pl.legend()
#pl.hold(True)
pl.show()













