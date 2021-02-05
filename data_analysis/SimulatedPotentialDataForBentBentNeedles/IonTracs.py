import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator
from scipy.fftpack import fft

class trajectory:
    """
    Class for numerical integration of the trajectory of a charged particle in a confining potential

    ubasis : 4 dimensional numpy array where first 3 dimensions store potential and last dimension encodes electrode number
    vrf_amp : RF voltage amplitude
    x,y,z : 1D arrays containing x,y,z values for potential specified in ubasis[:,:,:,i]. Default in microns
    rfelec_inphase : list with size same as number of electrodes (ubasis.shape[-1]). 1 for electrode, 0 for no electrode
    rfelec_out_phase : list with size same as number of electrodes (ubasis.shape[-1]). 1 for electrode, 0 for no electrode
    dcelec : list with size same as number of electrodes (ubasis.shape[-1]) with DC voltages on each electrode

    """
    def __init__(self,ubasis,x,y,z,vrf_amp,vrf_freq,rfelec_in_phase=None,rfelec_out_phase=None,
                 dcelec=None,mass_number = 171, charge = 1, **kwargs):
        self.ubasis = ubasis
        self.vrf_amp = vrf_amp
        self.vrf_freq = vrf_freq
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.omega = 2*np.pi*vrf_freq
        self.rfelec_inphase = np.array(rfelec_in_phase)
        self.rfelec_out_phase = np.array(rfelec_out_phase)
        self.dcelec = np.array(dcelec)
        self.mass_number = mass_number
        self.charge = charge
        self.q = self.charge * 1.60217662e-19 # multiplying by charge of electron in coulomb
        self.m = self.mass_number * 1.66054e-27 # multiplying by 1 AMU in Kg

        self.__dict__.update(kwargs)

    def GetEfieldInterpolators(self):
        """
        adds 3 lists to the attributes of the class
        Ex_interpolators : ith element in the list contains an interpolator for the X component of E field for the ith electrode
        Ey_interpolators : ith element in the list contains an interpolator for the Y component of E field for the ith electrode
        Ez_interpolators : ith element in the list contains an interpolator for the Z component of E field for the ith electrode

        """
        electrodes = self.ubasis.shape[-1] # number of electrodes


        Ex = np.empty(self.ubasis.shape)
        Ey = np.empty(self.ubasis.shape)
        Ez = np.empty(self.ubasis.shape)

        Interpolate_Ex = []
        Interpolate_Ey = []
        Interpolate_Ez = []

        for i in range(0,electrodes):
            Ex[:,:,:,i] , Ey[:,:,:,i] , Ez[:,:,:,i] = np.gradient(self.ubasis[:,:,:,i],self.x,self.y,self.z)

        for i in range(0,electrodes):
            Interpolate_Ex.append( RegularGridInterpolator( (self.x,self.y,self.z),-Ex[:,:,:,i] ) ) # negative sign because of gaus's law
            Interpolate_Ey.append( RegularGridInterpolator( (self.x,self.y,self.z),-Ey[:,:,:,i] ) )
            Interpolate_Ez.append( RegularGridInterpolator( (self.x,self.y,self.z),-Ez[:,:,:,i] ) )

            self.__dict__.update(Ex_interpolators = Interpolate_Ex, Ey_interpolators = Interpolate_Ey, Ez_interpolators = Interpolate_Ez)



    def GetTrajectory(self,r_0,t_values,noreturn=True,rotate_xy=True):
        """
        simulates the trajectory for a charged particle
        :parameter r_0 stands for the initial condition [x,y,z,vx,vy,vz] where (x,y,z) is position and (vx,vy,vz) is velocity
        :parameter t_values values of time over which the trajectory is to be calculated

        :return: array of size ( N X 6) where N is the length of t_values and 6 values correspond to position and velocity

        """

        if 'Ex_interpolators' in self.__dict__.keys(): # check if interpolators exist
            pass
        else:
            self.GetEfieldInterpolators()

        def R_dot(r,t):
            """
            Function to calculate time derivative of the vector containing position and velocity
            to be passed to odeint()
            :param r: [x,y,z,vx,vy,vz] where (x,y,z) is particle position and (vx,vy,vz) is  particle velocity
            :param t: time
            :return: [vx,vy,vz,ax,ay,az] where (vx,vy,vz) is  particle velocity and (ax,ay,az) is particle acceleration
            """

            xx = r[0]
            yy = r[1]
            zz = r[2]
            vx = r[3]
            vy = r[4]
            vz = r[5]

            # create array with voltage values on each electrode
            velec = self.vrf_amp * np.sin(self.omega* t) * (self.rfelec_inphase - self.rfelec_out_phase) + self.dcelec



            # # create list with interpolated values for Ex, Ey,Ez
            Ex_values = np.array( [interpolator([xx,yy,zz])[0] for interpolator in self.Ex_interpolators] )
            Ey_values = np.array( [interpolator([xx,yy,zz])[0] for interpolator in self.Ey_interpolators] )
            Ez_values = np.array([interpolator([xx, yy, zz])[0] for interpolator in self.Ez_interpolators] )

            #  find the acceleration ax,ay,az in X,Y,Z direction using the principle of superposition

            ax = np.dot(velec,Ex_values) * (self.q/self.m)
            ay = np.dot(velec,Ey_values) * (self.q/self.m)
            az = np.dot(velec,Ez_values) * (self.q/self.m)



            return [vx,vy,vz,ax,ay,az]

        # solving for the trajectory using odeint

        X_sol = odeint(R_dot,r_0,t_values)

        if rotate_xy == False:

            self.__dict__.update(solved_traj = X_sol)
            self.__dict__.update(t_values = t_values)
            self.__dict__.update(x_traj = X_sol[:,0])
            self.__dict__.update(y_traj = X_sol[:,1])
            self.__dict__.update(z_traj = X_sol[:,2])

        else :
            self.__dict__.update(solved_traj=X_sol)
            self.__dict__.update(t_values=t_values)
            self.__dict__.update(x_traj = (1/np.sqrt(2)) * (X_sol[:,0]+X_sol[:,1]))
            self.__dict__.update(y_traj =(-(1/np.sqrt(2)) * X_sol[:,0] + (1/np.sqrt(2)) * X_sol[:,1]))
            self.__dict__.update(z_traj=X_sol[:, 2])


        if noreturn:
            pass
        else:
            return X_sol


    def GetFrequencySpectrum(self,noreturn = False):
        """
        Calculates the frequency spectrum of the x,y,z trajectories. Requires running of GetTrajectory().
        Stores the frequencies in freq_axis attribute of the trajectory class
        Stores the x,y,z amplitude spectrum in ampl_x, ampl_y, ampl_z attribute of the trajectory class

        :return: None
        """

        if 'x_traj' in self.__dict__.keys():
            pass
        else:
            print(' Solve the trajectory first using GetTrajectory')
            return None

        def frequency_axis(time):
            N = time.size
            tstep = time[1] - time[0]
            if N % 2 == 0:
                freq_axis = np.linspace(0, 1 / (2 * tstep), N // 2)
            else:
                freq_axis = np.linspace(0, 1 / (2 * tstep), N // 2 + 1)
            return freq_axis

        def ampl_spectrum(fftx, freq_axis):
            fft_amps = (2 / fftx.size) * np.abs(fftx[0:freq_axis.size])
            return fft_amps


        fftx = fft(self.x_traj)
        ffty = fft(self.y_traj)
        fftz = fft(self.z_traj)

        freq_axis = frequency_axis(self.t_values)
        ampl_x = ampl_spectrum(fftx,freq_axis)
        ampl_y = ampl_spectrum(ffty,freq_axis)
        ampl_z = ampl_spectrum(fftz,freq_axis)

        self.__dict__.update(freq_axis = freq_axis)
        self.__dict__.update(ampl_x = ampl_x)
        self.__dict__.update(ampl_y = ampl_y)
        self.__dict__.update(ampl_z = ampl_z)

        if noreturn == False:
            return {'freq_axis': freq_axis, 'x_amlpitude' : ampl_x, 'y_amplitude' : ampl_y, 'z_ampl' : ampl_z}




