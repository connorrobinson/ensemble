import numpy as np
from astropy.io import ascii, fits
import photutils
import scipy.signal
import copy
from astropy.stats import sigma_clipped_stats, mad_std
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
import functools
from astropy.table import Table, Column
import pickle
import copy


##############################
# INDEPENDANT FUNCTIONS         
##############################

def euclid(x0, y0, x1, y1):
    '''
    euclid

    PURPOSE:
        Calculate the euclidian distance from a point x0, y0 for a different point, x1, y1

    INPUTS:
        x0:[float] First x coordinate in data units
        y0:[float] First y coordinate in data units
        x1: [float] Second x coordinate in data units
        y1: [float] Second y coordinate in data units

    AUTHOR:
        Connor Robinson, Feb 15th, 2021
    '''

    return np.sqrt((x1 - x0)**2 + (y1-y0)**2)

def displayEuclid(axis, x0, y0, x1, y1):
    '''
    
    PURPOSE:
        Calculate the euclidian distance from a point x0, y0 for a different point, x1, y1 in figure display unit space.
        x0,y0 or x1,y1 can be a set of coordinates (but not both)
    
    INPUTS:
        axis:[matplotlib axis] Axis for which to calculate display coordinates
        x0:[float/float arr] First x coordinate in data units
        y0:[float/float arr] First y coordinate in data units
        x1: [float/float arr] Second x coordinate in data units
        y1: [float/float arr] Second y coordinate in data units
        
    AUTHOR:
        Connor Robinson, Feb 18th, 2021
    '''
    
    #Convert from data coordinates to display coordinates
    #Do this in a loop because weird things happen otherwise.
    
    x0arr = np.atleast_1d(x0)
    y0arr = np.atleast_1d(y0)
    x1arr = np.atleast_1d(x1)
    y1arr = np.atleast_1d(y1)
    
    xy0 = np.array([axis.transData.transform((x0arr[i], y0arr[i])) for i in np.arange(len(x0arr))])
    xy1 = np.array([axis.transData.transform((x1arr[i], y1arr[i])) for i in np.arange(len(x1arr))])
    
    return euclid(xy0[:,0], xy0[:,1], xy1[:,0], xy1[:,1])


def loadEnsemble(name):
    '''

    PURPOSE:
        Load a pickle file containing an ensemble object.

    INPUTS:
        name:[str] Name of the object (does not include '.pkl')

    AUTHOR:
        Connor Robinson, Feb 19th, 2021

    '''
    return pickle.load(open(name+'.pkl', "rb" ))  


def update_errorbar(errobj, x, y, xerr=None, yerr=None):
    '''
    
    update_errorbar
    
    PURPOSE:
        Update errorbars after they have been plotted (for use with interactive plotting)
        
        
    INPUTS:
        errobj:[errorbar container] Errorbars to be changed
        x:[float array] New x values 
        y:[float arrau] New y values
    
    OPTIONAL INPUTS:
        xerr:[float array] New xerr values
        yerr:[float array] New yerr values
    
    AUTHOR:
        Source: https://stackoverflow.com/questions/25210723/matplotlib-set-data-for-errorbar-plot
    
    '''
    ln, caps, bars = errobj
    if len(bars) == 2:
        assert xerr is not None and yerr is not None, "Your errorbar object has 2 dimension of error bars defined. You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass
    ln.set_data(x,y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    try:
        barsx.set_segments([np.array([[xt, y], [xb, y]]) for xt, xb, y in zip(x + xerr, x - xerr, y)])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    try:
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y + yerr, y - yerr)])
    except NameError:
        pass


##############################
# ens CLASS     
##############################

class ens():
    '''
    
    ensemble
    
    PURPOSE:
        Object to be used to do ensemble photometry following Honeycutt (1992)
        
    AUTHOR:
        Connor Robinson, February 9th, 2021
    
    '''
    
    def __init__(self, photfiles, time, snr_cutoff = 0, save_name = 'ensemble', save_path = './'):
        '''
        
        __init__
        
        PURPOSE:
            Intialization function for the ensemble photometry class.
            
        INPUTS:
            photfiles:[string array] Array of photometry files. Those files must include the following columns:
                'aper_sum_bkgsub': Counts within an aperture after applying background subtraction.
                'err': Estimate of the uncertainty of the flux measurement. 
            time:[float array] An array of dates for each photometry file.
        
        OPTIONAL INPUTS:
            snr_cutoff:[float] SNR cutoff for excluding points.
        
        AUTHOR:
            Connor Robinson, February 10th, 2021
        '''
        
        
        #Add the photometry files, date, snr cutoff, and name to object
        self.photfiles = photfiles
        self.time = time
        self.snr_cutoff = snr_cutoff
        self.save_name = save_name
        self.save_path = save_path
        
        #Create empty arrays to fill later
        self.mask = []
        self.star = {}
        self.mw = []
        
        #Construct the mw array without any masking.
        self.phot = [ascii.read(pf) for pf in photfiles]
        
        #Get the number of exposures & number of stars
        self.ee = len(self.phot)
        self.ss = len(self.phot[0])
        
        self.counts = np.array([pt['aper_sum_bkgsub'] for pt in self.phot])
        self.err = np.array([pt['err'] for pt in self.phot])
        self.snr = self.counts/self.err
    
        #Calculate the instrumental magnitudes and propagate the uncertainty
        self.imag = -2.5 * np.log10(self.counts)
        self.imag_err = 0.434 * self.err/self.counts
        
        #Create m and w without any excluded stars or exposures        
        self.resetmw()
        
        return
    
    def resetmw(self):
        '''
        resetMask():
        
        PURPOSE:
            Reset the m and w arrays back to their initial state.
        
        INPUTS:
            None.
        
        AUTHOR:
            Connor Robinson, February 9th, 2021
        
        '''
        
        #Instrumental magnitude array
        self.m = -2.5 * np.log10(self.counts)
        
        #w1: Excluded exposures
        self.w1 = np.ones([self.ee]).astype('bool')
        
        #w2: Excluded stars
        self.w2 = np.ones([self.ss]).astype('bool')
        
        #w3: Excluded stars in specific exposures. This is NOT a boolean array (since it will be multiplied against w4)
        self.w3 = np.ones([self.ee, self.ss])
        
        #w4: Weights on the points
        self.w4 = 1/self.imag_err**2
        
        
    
    def solveEnsemble(self):
        '''

        PURPOSE:
            Construct and solve the matrices representing the linear set of equations to do differential photometry with missing points.
            
        INPUTS:
            None.

        AUTHOR:
            Connor Robinson, Dec, 17th, 2020
        '''
        
        #Construct new m and w matrices while excluding the missing stars and exposures.
        m = self.m[self.w1, :]
        m = m[:, self.w2]
        
        w3 = self.w3[self.w1, :]
        w3 = w3[:,self.w2]
        
        w4 = self.w4[self.w1, :]
        w4 = w4[:,self.w2]
        
        #Combine the final two sets of weights into the full w array. Save the result to the object.
        w = w3 * w4
        self.w = w
        
        #Define the new size of the matrix
        ee, ss = np.shape(w)
        
        #Construct the necessary matrices 
        A = []

        #Build the first part of the matrix (rows 0 to ee)
        for i in np.arange(ee):
            #Construct the row
            row = np.zeros(ee+ss)

            #Add the values
            row[i] = np.sum(w[i,:])
            row[ee:] = w[i,:]

            A.append(row)

        #Build the second part of the matrix (rows ee to ee + ss)
        for i in np.arange(ss):

            #Construct the row
            row = np.zeros(ee+ss)

            #Add the values
            row[:ee] = w[:,i]
            row[ee+i] = np.sum(w[:,i])

            A.append(row)

        A = np.array(A)

        #Build the array on the right side of the equation
        B = []

        #Go over each exposure
        for i in np.arange(ee):
            Bi = np.sum(w[i,:]*m[i,:])
            B.append(Bi)

        #Go through each star
        for i in np.arange(ss):
            Bi = np.sum(w[:,i]*m[:,i])
            B.append(Bi)

        #Solve the system of linear equations
        x = np.linalg.solve(A, B)

        #Break up the solution vector into its components, em (exposure magnitude for exposure e) and m0 (mean instrumental magnitude of the star without transparency variations)
        em = x[:ee]
        m0 = x[ee:]

        #Calculate the differential magnitudes
        M = (m.T - em)

        #Get the errors on m0
        var_m0 = []
        for i in np.arange(ss):
            var_m0_i = ee * np.sum( (m[:,i] - em - m0[i])**2 * w[:,i] )/((ee-1) * np.sum(w[:,i]))
            var_m0.append(var_m0_i)
        var_m0 = np.array(var_m0)
        err_m0 = np.sqrt(var_m0)

        #Get the errors on em
        var_em = []
        for i in np.arange(ee):
            var_em_i = ss * np.sum( (m[i,:] - em[i] - m0[:])**2 * w[i,:])/( (ss-1) * np.sum(w[i,:]) )
            var_em.append(var_em_i)
        var_em = np.array(var_em)
        err_em = np.sqrt(var_em)
        
        #Get the errors for the photometry values
        var_m = (self.imag_err**2)[self.w1, :]
        var_m = var_m[:,self.w2]
        
        var_M = var_m.T + var_em
        
        #Normalize the instrumental magnitudes with a uniform offset based on the exposure with the lowest opacity
        self.offset = np.nanmin(em)
        em = em - self.offset
        
        
        #Return these values back into a full array with nans for missing exposures/stars
        self.em = np.zeros(self.ee) + np.nan
        self.m0 = np.zeros(self.ss) + np.nan
        self.err_m0 = np.zeros(self.ss) + np.nan
        self.err_em = np.zeros(self.ee) + np.nan
        
        self.em[np.where(self.w1)[0]] = em
        self.m0[np.where(self.w2)[0]] = m0 + self.offset
        self.err_em[np.where(self.w1)[0]] = err_em
        self.err_m0[np.where(self.w2)[0]] = err_m0
        
        
        self.M = (self.imag.T - self.em).T
#         self.err_M = np.sqrt((self.imag_err**2).T + self.err_em**2).T
        
        self.ss_eff = np.sum(self.w3, axis = 1)
    
        self.err_M = np.sqrt(self.imag_err.T**2 + self.err_em**2/self.ss_eff).T
    
        return
    
    def basicExclude(self):
        '''
        
        basicExcluse
        
        PURPOSE:
            Exclude the star in specific exposures where someone is clearly wrong.
            This includes negative counts, nans, and a SNR cutoff.
            Adds any exposures that are missing all stars to w1, and stars that are missing from all exposures to w2
            
        INPUTS:
            None
        
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        
        #Replace imag of stars with negative/nan counts with a large number
        self.m[~np.isfinite(self.imag)] = 1e15
        
        #Exclude the specific star/exposure pairs with negative/nan counts
        self.w3[~np.isfinite(self.imag)] = 0
        
        #If a snr cutoff is used, apply that here.
        self.snrExclude(self.snr_cutoff)
        
        #Assign 0 weight to points with nans for weights
        self.w3[~np.isfinite(self.w4)] = False
        self.w4[~np.isfinite(self.w4)] = 0
        
        #If a star is missing in ALL exposures, add it to the excluded stars in w2
        self.excludeMissingStars()
        
        #If ALL stars are missing in a single exposure, add it to the excluded exposures in w1        
        self.excludeMissingExposures()
        
        
        
    def plotEnsemble(self, index = None, indicator_size = 3):
        '''
        PURPOSE:
            Plots the results from ensembleSolver

        INPUTS:
            None

        OPTIONAL INPUTS:
            index: Index for an individual light curve
            markersize:[float] Size of the indicators for whether an exposure is excluded or not 


        AUTHOR:
            Connor Robinson, Dec, 17th, 2020
        '''
        
        #Handle a few things regarding selecting an individual star for plotting
        self.index = index
        self.choose_star = False
        
        #Make plots of everything
        fig, ax = plt.subplots(2,4,figsize = [15,7])
        plt.subplots_adjust(hspace = 0.3, wspace = 0.35)
        
        #Define each axis
        self.ax1 = ax[0,0]
        self.ax2 = ax[0,1]
        self.ax3 = ax[0,2]
        self.ax4 = ax[0,3]
        self.ax5 = ax[1,0]
        self.ax6 = ax[1,1]
        self.ax7 = ax[1,2]
        self.ax8 = ax[1,3]
        
        
        # AXIS 1
        ########################################
        #Exposure magnitude vs. exposure
        self.m_vs_exp = self.ax1.errorbar(np.arange(self.ee), self.em, yerr = self.err_em, color = 'k', ecolor = 'r', marker = 'o', markersize = 2)
        
        #Create a set of blue/red points which indicate if the exposure is being used. These will be turned on/off when you click on them.
        self.x_exp = np.arange(self.ee)
        self.y_exp = np.zeros(self.ee)+np.nanmax(self.em)+0.5
        
        #Commas to avoid issues with objects being lists
        self.exp_yes, = self.ax1.plot(self.x_exp[self.w1], self.y_exp[self.w1], color = 'b', marker = 's', ls = '', markersize = indicator_size)
        self.exp_no, = self.ax1.plot(self.x_exp[~self.w1], self.y_exp[~self.w1], color = 'r', marker = 's', ls = '', markersize = indicator_size)
        
        self.ax1.set_xlabel('Exposure', fontsize = 12)
        self.ax1.set_ylabel(r'Exposure Magnitude, $em$', fontsize = 12)
        self.ax1.set_ylim([np.nanmax(self.em)+1, np.nanmin(self.em)-1])
        
        
        # AXIS 2 -- Exposure magnitude uncertainty vs. exposure
        ########################################
        self.merr_vs_exp, = self.ax2.plot(np.arange(self.ee), self.err_em, color = 'k', marker = 'o', markersize = 2)
        self.ax2.set_xlabel('Exposure', fontsize = 12)
        self.ax2.set_ylabel(r'$\sigma_{em}$', fontsize = 12)
        self.ax2.set_ylim([-0.15, np.nanmax(self.err_em)+0.1])
        
        #Create a set of blue/red points which indicate if the exposure is being used. These will be turned on/off when you click on them.
        self.y_exp_err = np.zeros(self.ee) - 0.1
        self.exp_err_yes, = self.ax2.plot(self.x_exp[self.w1], self.y_exp_err[self.w1], color = 'b', marker = 's', ls = '', markersize = indicator_size)
        self.exp_err_no, = self.ax2.plot(self.x_exp[~self.w1], self.y_exp_err[~self.w1], color = 'r', marker = 's', ls = '', markersize = indicator_size)

        # AXIS 3 -- m0 vs sigma m0
        ########################################
        self.m0_vs_m0err, = self.ax3.plot(self.m0, self.err_m0, ls = '', marker = 'o', color = 'k', alpha = 0.5)
        self.ax3.set_xlabel('Mean Instrumental Magnitude without \n '+r'transparency variations, $m_0$', fontsize = 12)
        self.ax3.set_ylabel(r'$\sigma_{m_0}$', fontsize = 12);
        self.ax3.set_ylim([-0.05, np.nanmax(self.err_m0)+0.1])
        self.ax3.set_xlim([np.nanmin(self.m0)-0.1, np.nanmax(self.m0)+0.1])    

        #Highlight the selected star in blue (if one is selected)
        self.mark_selected, = self.ax3.plot([],[], color = 'blue', marker = 'o', alpha = 0.7, ls = '')
        
        #Mark the stars that are to be excluded. This should not actually plot anything.
        self.star_select = np.zeros(self.ss).astype('bool')
        self.star_excl, = self.ax3.plot([], [], color = 'r', marker = 'o', ls = '', zorder = 2, alpha = 0.5)
        
        #Mark the stars that have been de-weighted. This will not plot anything at first.
        self.orig_w4_weight = np.ones(self.ss).astype('bool')
        self.deweight, = self.ax3.plot([],[], markerfacecolor = 'None', markeredgecolor = 'purple', marker = 'o', ls = '', zorder = 2, alpha = 0.5, markersize = 20)
        
        # AXIS 4 -- em vs sigma em
        ########################################
        self.em_vs_emerr, = self.ax4.plot(self.em, self.err_em, ls = '', marker = 'o', color = 'k', alpha = 0.5)
        self.ax4.set_xlabel(r'Exposure Magnitude, $em$', fontsize = 12)
        self.ax4.set_ylabel(r'$\sigma_{em}$', fontsize = 12);
        
        #Mark the exposures that are to be excluded. This should not actually plot anything.
        self.exp_select = np.zeros(self.ee).astype('bool')
        self.exp_excl, = self.ax4.plot([], [], color = 'r', marker = 'o', ls = '', zorder = 2, alpha = 0.5)
        self.ax4.set_ylim([-0.05, np.nanmax(self.err_em)+0.2])
        self.ax4.set_xlim([np.nanmin(self.em)-1, np.nanmax(self.em)+1])
        
        # AXIS 5 -- Histogram of numbers of exposures used for each star
        ########################################
        used = self.w > 0
        nexp = np.sum(used, axis = 0)
        nstar = np.sum(used, axis = 1)

        self.nstar_hist = self.ax5.hist(nstar, bins = self.ss, range = [0,self.ss], color = 'k', alpha = 0.5)
        self.ax5.set_xlim([0, self.ss])
        self.ax5.set_xlabel(r'Number of stars, $N_{s}$', fontsize = 12)
        self.ax5.set_ylabel('Number of exposures\n'+r'containing $N_{s}$ stars', fontsize = 12)
        
        # AXIS 6 -- histogram of the number of stars appearing in all N_exp exposures 
        ########################################
        self.nexp_hist = self.ax6.hist(nexp, bins = self.ee, range = [0,self.ee], color = 'k', alpha = 0.5, align = 'left')
        self.ax6.set_xlim([-0.5, self.ee])
        self.ax6.set_xlabel('Number of Exposures, $N_{e}$', fontsize = 12)
        self.ax6.set_ylabel('Number of stars appearing \n'+r'in all $N_{e}$ exposures', fontsize = 12);
    
        
        # AXIS 7 -- Instrumental magntiude vs. exposure number
        ########################################        
        
        clin = np.linspace(0,1,self.ss)
        self.M_vs_exp = []
        
        for i, st in enumerate(self.M.T):
            self.M_vs_exp.append(self.ax7.plot(st, color = cm.gist_rainbow(clin[i]), alpha = 0.5, marker = 'o', markersize = 5))
        self.ax7.set_xlabel('Exposure', fontsize = 12)
        self.ax7.set_ylabel(r'Instrumental Magnitude, $m$', fontsize = 12)
        self.ax7.set_ylim([np.nanmax(self.M[self.M != np.inf])+1, np.nanmin(self.M[self.M != -np.inf])-1]);
        
        # AXIS 8 light curve for an individual object
        ########################################
        if self.index != None:
            self.buildLightCurve()
            self.ind_lc = self.ax8.errorbar(self.lct, self.lcM, yerr = self.lc_err_M, marker = 'o', color = 'k', ls = '', alpha = 0.2, ms = 3, ecolor = 'r')
            
            self.ax8.set_xlabel('Time [d]')
            self.ax8.set_ylabel('Instrumental Magnitude, m');
            
            self.ax8.set_ylim([np.nanmax(self.lcM)+1, np.nanmin(self.lcM)-1]);
            self.mark_selected.set_data(self.m0[self.index], self.err_m0[self.index])
            
            #Create markers for excluded points.
            self.lc_excl = self.ax8.plot([],[], marker = 'o', color = 'r', ls = '', alpha = 0.5, ms = 3)
            self.lc_ind_label = self.ax8.text(0.97, 0.97, 'index:'+str(self.index), transform = self.ax8.transAxes, ha = 'right', va = 'top')
            
        else:
            self.ax8.set_visible(False)     
        
        #Connect the canvas for mouse clicks
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        
        ######################################
        #BUTTONS
        ######################################
        
        #Add a button widget to recalculate the solution and update the figures.
        solve_ensemble_button = widgets.Button(description='Re-calculate Solution')
        output = widgets.Output()
        
        def on_solve_ensemble_button_clicked(b, self = self):            
            
            #Catch missing stars/exposures
            self.excludeMissingExposures()
            self.excludeMissingStars()
            
            #Solve the ensemble again
            self.solveEnsemble()
            
            #Clear the selected stars
            self.star_select = np.zeros(self.ss).astype('bool')
            
            #Clear the selected exposures 
            self.exp_select = np.zeros(self.ee).astype('bool')
            
            #Rebuild the light curve if an object is selected.
            if self.index != None:
                self.buildLightCurve()
            
            
            #Update the figures
            self.update_figure()
            
        solve_ensemble_button.on_click(functools.partial(on_solve_ensemble_button_clicked, self=self))
        
        #Add a button widget to reset all of the weights
        reset_weights_button = widgets.Button(description='Reset weights')
        
        def on_reset_weights_button_clicked(b, self = self):
            #Solve the ensemble again
            self.resetWeights()
            
            #Reset the markers to show that weights that have been modified
            self.orig_w4_weight = np.ones(self.ss).astype('bool')
            
            #Apply the basic exclusion again.
            self.basicExclude()
            
            #Update the figures
            self.update_figure()
            

        reset_weights_button.on_click(functools.partial(on_reset_weights_button_clicked, self=self))
        
        #Add a button widget to select a star from panel 3 for plotting
        select_star_button = widgets.Button(description='Select Star from Panel 3')        
        
        def on_select_star_button_clicked(b, self = self):
            
            #Clear any currently selected stars by replotting the figure
            self.update_figure()
            
            #Turn on the choose_star flag (used by __call__)
            self.choose_star = True
            
        select_star_button.on_click(functools.partial(on_select_star_button_clicked, self=self))
        
        
        #Add a button widget to select a star to re-weight.
        weight_star_button = widgets.Button(description='Remove/add weight to star')        
        
        def on_weight_star_button_clicked(b, self = self):
            
            #Clear any currently selected stars by replotting the figure
            self.update_figure()
            
            #Turn on the choose_star flag (used by __call__)
            self.weight_star = True
            
        weight_star_button.on_click(functools.partial(on_weight_star_button_clicked, self=self))
        
        
        ######################################
        # SAVE BUTTON
        ######################################
        #Add a button to save a complete version of the object as a pickle, a .dat file containing the final mask (w1*w2*w3*w4) and a .dat file containg the LC. 
        
        self.save_string = widgets.Text(value = self.save_name, placecolder = 'name', description= 'Save name:', disabled=False)
        save_button = widgets.Button(description='Save Weights/LC')
        
        def on_save_button_clicked(b, self = self):
            
            #Update the object save name with the vlue in the entry box
            self.save_name = self.save_string.value
            
            #Save the LC and the weights.
            if self.index != None:
                self.saveLC()
            
            self.saveMask()
            
            return
        
        
        save_button.on_click(functools.partial(on_save_button_clicked, self=self))
        
        #Place the buttons into a box and display it
        box =widgets.HBox([solve_ensemble_button, reset_weights_button, select_star_button, weight_star_button, self.save_string, save_button])
        display(box, output)
        
        fig.canvas.draw_idle()
        
        return
        
    def __call__(self, event):
        '''
        
        __call__
        
        PURPOSE:
            Handles clicks inside of plots
        
        AUTHOR:
            Connor Robinson, Dec, 17th, 2020
        '''        
        self.x = event.xdata
        self.y = event.ydata
        
        if (event.inaxes != self.ax1) \
        and (event.inaxes != self.ax2) \
        and (event.inaxes != self.ax3) \
        and (event.inaxes != self.ax4) \
        and (event.inaxes != self.ax8):
            return
        

        elif event.inaxes == self.ax1:
            
            #Calculate the euclidian distanc to find the closest point in display coordinates
            euc = displayEuclid(self.ax1, self.x, self.y, self.x_exp, self.y_exp)
            selected = np.nanargmin(euc)
            
            if self.w1[selected] == True:
                self.excludeExposure(selected)
            else:
                self.includeExposure(selected)
        
        elif event.inaxes == self.ax2:
            
            #Calculate the euclidian distanc to find the closest point.
            euc = displayEuclid(self.ax2, self.x, self.y, self.x_exp, self.y_exp_err)
            
            
            selected = np.nanargmin(euc)
    
            if self.w1[selected] == True:
                self.excludeExposure(selected)
            else:
                self.includeExposure(selected)
        
            
        elif event.inaxes == self.ax3:
            euc = displayEuclid(self.ax3, self.x, self.y, self.m0, self.err_m0)
            selected = np.nanargmin(euc)
            
            if self.choose_star == True:
                #Turn of the flag
                self.choose_star = False
            
                #Calculate the euclidian distanc to find the closest point.
                euc = displayEuclid(self.ax3, self.x, self.y, self.m0, self.err_m0)
                selected = np.nanargmin(euc)
            
                self.index = selected
                self.buildLightCurve()
            
            elif self.weight_star == True:
                
                #Turn of the flag
                self.weight_star = False
                
                #Calculate the euclidian distanc to find the closest point.
                euc = displayEuclid(self.ax3, self.x, self.y, self.m0, self.err_m0)
                selected = np.nanargmin(euc)
                
                
                if self.orig_w4_weight[selected] == True:
                    self.deweightStar(selected)
                    self.orig_w4_weight[selected] = False
                else:
                    self.reweightStar(selected)
                    self.orig_w4_weight[selected] = True
                
            
            else:
                if self.star_select[selected] == True:
                    self.star_select[selected] = False
                    self.includeStar(selected)
                    
                    self.lct = None
                    self.lcM = None
                    self.lce = None
                    self.lc_select = np.array([]).astype('bool')
                    
                elif self.star_select[selected] == False:
                    self.star_select[selected] = True
                    self.excludeStar(selected)
                

        elif event.inaxes == self.ax4:
            euc = displayEuclid(self.ax4, self.x, self.y, self.em, self.err_em)
            selected = np.nanargmin(euc)
            
            if self.exp_select[selected] == True:
                self.exp_select[selected] = False
                self.includeExposure(selected)
                
            elif self.exp_select[selected] == False:
                self.exp_select[selected] = True
                self.excludeExposure(selected)
        
        
        elif event.inaxes == self.ax8 and self.index != None:
            #Exclude specific stars from the light curve
            euc = displayEuclid(self.ax8, self.x, self.y, self.lct, self.lcM)            
            selected = np.nanargmin(euc)
            
            if self.lc_select[selected] == True:
                self.lc_select[selected] = False
                self.includeSpecific(self.index, self.lce[selected])
                
            elif self.lc_select[selected] == False:
                self.lc_select[selected] = True
                self.excludeSpecific(self.index, self.lce[selected])
            
        #Update the figure
        self.update_figure()    
        self.fig.canvas.draw()
        
        return
    
    
    def update_figure(self):
        '''
        self.update_figure()
        
        PURPOSE:
            Update the figure after any changes have been made via clicks/refreshes. 
        
        INPUTS:
            None
        
        AUTHOR:
            Connor Robinson, Feb 17th, 2021
        
        '''
        
        # UPDATE AX 1
        update_errorbar(self.m_vs_exp, np.arange(self.ee), self.em, xerr=None, yerr=self.err_em)
        self.ax1.set_ylim([np.nanmax(self.em)+1, np.nanmin(self.em)-1])

        #Adjust the indicators
        self.y_exp = np.zeros(self.ee)+np.nanmax(self.em)+0.5
        self.exp_yes.set_data(self.x_exp[self.w1], self.y_exp[self.w1])
        self.exp_no.set_data(self.x_exp[~self.w1], self.y_exp[~self.w1])

        # UPDATE AX 2
        self.merr_vs_exp.set_data(np.arange(self.ee), self.err_em)
        self.ax2.set_ylim([-0.15, np.nanmax(self.err_em)+0.1])
        
        #Adjust the indicators
        self.exp_err_yes.set_data(self.x_exp[self.w1], self.y_exp_err[self.w1])
        self.exp_err_no.set_data(self.x_exp[~self.w1], self.y_exp_err[~self.w1])
        
        # UPDATE AX 3
        self.m0_vs_m0err.set_data(self.m0, self.err_m0)
        self.ax3.set_ylim([-0.05, np.nanmax(self.err_m0)+0.1])
        self.ax3.set_xlim([np.nanmin(self.m0)-0.1, np.nanmax(self.m0)+0.1])    
        
        self.star_excl.set_data(self.m0[self.star_select], self.err_m0[self.star_select])
        self.deweight.set_data(self.m0[~self.orig_w4_weight], self.err_m0[~self.orig_w4_weight])
        
        # UPDATE AX 4
        self.em_vs_emerr.set_data(self.em, self.err_em)
        self.exp_excl.set_data(self.em[self.exp_select], self.err_em[self.exp_select])
        self.ax4.set_ylim([-0.05, np.nanmax(self.err_em)+0.2])
        self.ax4.set_xlim([np.nanmin(self.em)-1, np.nanmax(self.em)+1])
        
        if self.index != None:
            self.mark_selected.set_data(self.m0[self.index], self.err_m0[self.index])
        
        # UPDATE AX 5
        self.ax5.cla()
        used = self.w > 0
        nexp = np.sum(used, axis = 0)
        nstar = np.sum(used, axis = 1)

        self.nstar_hist = self.ax5.hist(nstar, bins = self.ss, range = [0,self.ss], color = 'k', alpha = 0.5)
        self.ax5.set_xlim([0, self.ss])
        self.ax5.set_xlabel(r'Number of stars, $N_{s}$', fontsize = 12)
        self.ax5.set_ylabel('Number of exposures\n'+r'containing $N_{s}$ stars', fontsize = 12)
        
        # UPDATE AX 6
        self.ax6.cla()
        self.nexp_hist = self.ax6.hist(nexp, bins = self.ee, range = [0,self.ee], color = 'k', alpha = 0.5, align = 'left')
        self.ax6.set_xlim([-0.5, self.ee])
        self.ax6.set_xlabel('Number of Exposures, $N_{e}$', fontsize = 12)
        self.ax6.set_ylabel('Number of stars appearing \n'+r'in all $N_{e}$ exposures', fontsize = 12);
        
        #UPDATE AX 7
        self.ax7.cla()
        clin = np.linspace(0,1,self.ss)
        self.M_vs_exp = []
        for i, st in enumerate(self.M.T):
            self.M_vs_exp.append(self.ax7.plot(st, color = cm.gist_rainbow(clin[i]), alpha = 0.5, marker = 'o', markersize = 5))

        self.ax7.set_xlabel('Exposure', fontsize = 12)
        self.ax7.set_ylabel(r'Instrumental Magnitude, $m$', fontsize = 12)
        self.ax7.set_ylim([np.nanmax(self.M[self.M != np.inf])+1, np.nanmin(self.M[self.M != -np.inf])-1])
        
        #UPDATE AX 8
        if self.index != None:        
            self.ax8.set_visible(True)
            self.ax8.cla()
            
            self.ind_lc = self.ax8.errorbar(self.lct, self.lcM, yerr = self.lc_err_M, marker = 'o', color = 'k', ls = '', alpha = 0.2, ms = 3, ecolor = 'r')
            self.ax8.set_xlabel('Time [d]')
            self.ax8.set_ylabel('Instrumental Magnitude, m');
            self.ax8.set_ylim([np.nanmax(self.lcM)+1, np.nanmin(self.lcM)-1])
            self.mark_selected.set_data(self.m0[self.index], self.err_m0[self.index])
            
            #Update the excluded points
            self.lc_excl = self.ax8.plot(self.lct[self.lc_select], self.lcM[self.lc_select], marker = 'o', color = 'r', ls = '', alpha = 0.5, ms = 3)
            
            #Add the index
            self.lc_ind_label = self.ax8.text(0.97, 0.97, 'index:'+str(self.index), transform = self.ax8.transAxes, ha = 'right', va = 'top')
            
        
    def snrExclude(self, snr):
        '''
        snrExclude
        
        PURPOSE:
            Exclude stars based on a SNR cutoff
        
        '''
        #Apply a basic SNR cutoff if the snr keyword is used
        self.w3[self.snr < snr] = 0
        self.m[self.snr < snr] = 1e15
        
    
    def excludeMissingStars(self):
        '''
        PURPOSE:
            Exclude stars from w2 if they are entirely excluded by w3
            This should help the stability of the matrix solution (by excluding missing entire stars, rather than setting them to 0)
            
        INPUTS:
            None.
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        
        #Identify the missing stars
        inds = np.where(np.sum(self.w3, axis = 0) == 0)[0]
        
        self.excludeStar(inds)
    
    def excludeMissingExposures(self):
        '''
        PURPOSE:
            Exclude exposures from w1 if they are entirely excluded by w3
            This should help the stability of the matrix solution (by excluding missing entire exposures, rather than setting them to 0)
            
        INPUTS:
            None.
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        
        #Identify the missing stars
        inds = np.where(np.sum(self.w3, axis = 1) == 0)[0]
        
        self.excludeExposure(inds)
        
    
    def excludeExposure(self, ind):
        '''
        PURPOSE:
            Exclude an exposure from the matrix
        
        INPUTS:
            ind:[int] Index of the exposure to exclude (0 is the first exposure)
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        
        self.w1[ind] = False
#         self.w3[ind,:] = False
    
    def excludeStar(self, ind):
        '''
        PURPOSE:
            Exclude a star from the matrix
        
        INPUTS:
            ind:[int] Index of the star to exclude (0 is the first star)
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        self.w2[ind] = False
        self.w3[:,ind] = False
    
    def excludeSpecific(self, sind, eind):
        '''
        
        PURPOSE:
            Exclude specific stars from specific exposures
        INPUTS:
            sind:[int/int array] Indices of the stars to exclude (0 is the first star)
            eind:[int/int array] Indices of the exposures to include (0 is the first exposure)
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        self.w3[eind,sind] = False
        
    
    def includeExposure(self, ind):
        '''
        PURPOSE:
            Include an exposure from the matrix
        
        INPUTS:
            ind:[int] Index of the exposure to include (0 is the first exposure)
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        
        self.w1[ind] = True
    
    def includeStar(self, ind):
        '''
        PURPOSE:
            Include a star from the matrix
        
        INPUTS:
            ind:[int] Index of the star to include (0 is the first star)
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        self.w2[ind] = True
    
    def includeSpecific(self, sind, eind):
        '''
        
        PURPOSE:
            Include specific stars from specific exposures
        INPUTS:
            sind:[int/int array] Indices of the stars to exclude (0 is the first star)
            eind:[int/int array] Indices of the exposures to include (0 is the first exposure)
            
        AUTHOR:
            Connor Robinson, February 9th, 2021
        '''
        self.w3[eind,sind] = True
    
    
    def deweightStar(self, ind):
        '''
        
        PURPOSE:
            Set the weight of an individual star to a small, but non-zero number.
            This is useful in the case where your science target is bright.
            This is applied to all instances of this star in all exposures. 
            
        INPUTS:
            ind: Index of the star to de-weight (0 is the first star)
        
        '''
        
        self.w4[:,ind] = 1e-20
    
    def reweightStar(self, ind):
        '''
        
        PURPOSE:
            Restore the original weights of a star to their original values
            This is applied to all instances of this star in all exposures. 
            
        INPUTS:
            ind: Index of the star to de-weight (0 is the first star)
        
        '''
        
        self.w4[:,ind] = 1/self.imag_err[:,ind]**2
    
    def resetWeights(self):
        '''
        
        PURPOSE:
            Reset all of the weights
            
        INPUTS:
            None.
            
        AUTHOR:
            Connor Robinson, February 12th, 2021
        
        '''
        
        self.w1[:] = True
        self.w2[:] = True
        self.w3[:,:] = True
        self.w4 = 1/self.imag_err**2
        
        
    def buildLightCurve(self):
        '''
        
        PURPOSE:
            Construct a light curve from the solution for the current index and assign it to the lct (time), lcM (corrected instr. magnitude), lce (exposure index) attributes.
            Also clears any previously selected points
        
        INPUTS:
            None.
        
        AUTHOR:
            Connor Robinson, Feb 17th, 2021
        '''
        
        
        self.lct = self.time[self.w3[:,self.index].astype('bool')]
        self.lcM = self.M[self.w3[:,self.index].astype('bool'), self.index]
        self.lc_err_M = self.err_M[self.w3[:,self.index].astype('bool'), self.index]
        self.lce, = np.where(self.w3[:,self.index])
        self.lc_select = np.zeros(len(self.lct)).astype('bool')
    
    
    def saveLC(self):
        '''
        
        PURPOSE:
            Save a .dat file of the light curve
        
        
        INPUTS:
            None.
            
        AUTHOR:
            Connor Robinson, Feb 19th, 2021
        
        '''
        
        #Get the information for the source from each of the phot files
        pf_info = [list(p[self.index]) for p in self.phot]
        
        #Transpose the list (avoid going to array to maintain data types)
        pf_info_t = [list(i) for i in zip(*pf_info)]
        
        #Create a table based on the phot files info
        ens_table = Table(pf_info_t, names = self.phot[0].colnames)
        
        #Add columns to ens_table from the ensemble solution
        ens_table.add_column(Column(self.time, 'date'))
        ens_table.add_column(Column(self.M[:, self.index], 'imag'))
        ens_table.add_column(Column(self.err_M[:,self.index], 'err_imag'))
        ens_table.add_column(Column(self.em, 'em'))
        ens_table.add_column(Column(self.err_em, 'err_em'))
        
        ascii.write(ens_table, self.save_path + self.save_name+'.dat', overwrite = True)
        

        
    def saveMask(self):
        '''
        PURPOSE:
             Save the associated mask (w1 * w2 * w3)
             
        INPUTS:
            None:
            
            
        AUTHOR:
            Connor Robinson, Feb 19th, 2021
        
        '''
        #Get the weights in the right shape
        w1 = np.tile(self.w1, (self.ss, 1)).T
        w2 = np.tile(self.w2, (self.ee, 1))
        w1.shape, w2.shape, self.w3.shape
        
        #Create a master weight array
        w = w1 * w2 * self.w3
        
        w = w.astype(int)
        
        np.savetxt(self.save_path + self.save_name+'_mask.dat', w, fmt = '%i')
        
    
    def loadMask(self, maskfile = None):
        '''
        
        PURPOSE:
            Load in a previously created mask file and apply it to your light curve. 
            
            
        OPTIONAL INPUTS:
            maskfile:[str] File containing the w1 * w2 * w3 mask created by saveLC. If not specified, default will be:
                           self.save_path + self.save_name+'_mask.dat'
        
        
        AUTHOR:
            Connor Robinson, Mar 19th, 2021
        '''
        
        if maskfile == None:
            maskfile = self.save_path + self.save_name+'_mask.dat'
        
        mask = np.genfromtxt(maskfile)
        
        #Apply the mask and remove missing stars/exposures from w1 and w2. 
        self.w3 = mask
        self.excludeMissingStars()
        self.excludeMissingExposures()
        
    
        
    
        
#     def saveEnsemble(self):
#         '''
#        
#         NOTE!!!!! THIS IS DID NOT WORK. FUNDAMENTAL ISSUES WITH PICKLING AN OBJECT WITH OPEN PLOTS.
#    
#    
#         PURPOSE:
#             Save a pickle file containing the object.
        
#         INPUTS:
#             None.
            
#         AUTHOR:
#             Connor Robinson, Feb 19th, 2021
        
#         '''
        
#         #Create a copy of the pickle (without the figures)
#         ens_copy = copy.deepcopy(self)
        
#         return enscopy
        
#         pickle.dump(self, open(self.save_path+self.save_name+'.pkl', "wb" ))
        