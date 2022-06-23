# -*- coding: utf-8 -*-
# pysdvuln
# Copyright (C) 2021-2022 Salvatore Iacoletti
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""

import os
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.io import loadmat
from scipy.stats import gmean
import matplotlib.pyplot as plt
import eqsig
from myutils.utils_pickle import save_pickle, load_pickle


class Record():
    '''
    Ground-motion record container
    '''
    
    LAG_T_STEP = 0.1
    
    def __init__(self, gmr, t_step, unit="m/s2", **kwargs):
        '''
        gmr: ground motion record (changed to be always in m/s2 within Record)
        dt: time interval of gmr
        kwargs: other information if needed
        '''
        if unit not in ["g", "m/s2"]:
            raise Exception("unit can only be 'g' or 'm/s2'")
        gmr = self.check_gmr(gmr, unit)
        self.gmr = np.array(gmr)
        self.t_step = t_step
        self.pga = np.max([np.max(self.gmr), -np.min(self.gmr)])
        self.kwargs = kwargs
        self.__dict__.update(kwargs)
    
    
    def scale(self, scaling_factor):
        gmr = scaling_factor * self.gmr
        kwargs = deepcopy(self.kwargs)
        if "precomp_spectrum" in kwargs.keys():
            kwargs["precomp_spectrum"][:,1] = np.abs(scaling_factor) * \
                                              kwargs["precomp_spectrum"][:,1]
        return self.__class__(gmr, self.t_step, unit="m/s2", **kwargs)

    
    @classmethod
    def check_gmr(cls, gmr, unit):
        if len(gmr.shape) > 1:
            if gmr.shape[0] == 1:
                gmr = gmr[0,:]
            elif gmr.shape[1] == 1:
                gmr = gmr[:,0]
            else:
                raise Exception("the shape of gmr is not consistent, it should be a column vector!")
        if unit == "g":
            gmr = gmr*9.81 # make sure gmr is always in m/s2
        return gmr
    
    @property
    def num_points(self):
        return len(self.gmr)

    @property
    def duration(self):
        return (self.num_points-1) * self.t_step

    @property
    def time(self):
        # 1e-6 to avoid numerical errors
        return np.arange(0., (self.num_points-1e-6) * self.t_step, self.t_step)

    @property
    def arias_intensity(self):
        record = self.get_eqsig_record()
        return eqsig.im.calc_arias_intensity(record)
    
    
    def get_gmr(self, unit="m/s2"):
        if unit == "g":
            return self.gmr/9.81
        elif unit == "m/s2":
            return self.gmr
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")


    def get_time_gmr(self, unit="m/s2", resting_time=0.):
        if resting_time == 0.:
            return np.vstack([self.time, self.get_gmr(unit)]).T
        else:
            time_rest = np.arange(self.time[-1] + self.LAG_T_STEP,
                                  self.time[-1] + resting_time + self.LAG_T_STEP,
                                  self.LAG_T_STEP)
            return np.vstack([np.hstack([self.time, time_rest]),
                              np.hstack([self.get_gmr(unit), [0.]*time_rest.shape[0]])]).T


    def get_significant_duration(self, start=0.05, end=0.95, se=False):
        record = self.get_eqsig_record()
        return eqsig.im.calc_sig_dur(record, start=start, end=end, se=se)
    
    
    def get_eqsig_record(self):
        if "record" not in self.__dict__.keys():
            self.record = eqsig.AccSignal(self.gmr, self.t_step)
        return self.record
    

    def generate_response_spectrum(self, periods=None, add_pga=True):
        '''
        it returns Time (s) Acceleration (m/s2) in 2d array format
        '''
        # if "comp_spectrum" in self.__dict__.keys():
        #     return self.comp_spectrum
        record = self.get_eqsig_record()
        if periods is None:
            # compute the elastic response for 100 periods between T=0.01s and 10.0s
            periods = np.logspace(np.log10(0.01), np.log10(10), 100)
        if not hasattr(periods, '__iter__'):
            periods = np.array([periods])
        record.generate_response_spectrum(response_times=periods)
        times = record.response_times
        s_a = record.s_a
        if add_pga:
            times = np.hstack([[0.], times])
            s_a = np.hstack([[self.pga], s_a])
        self.comp_spectrum = np.array([times, s_a]).T # format 2d array
        return self.comp_spectrum


    def generate_acc_vel_disp_series(self):
        '''
        it returns Time (s) Acceleration (m/s2) Velocity (m/s) Displacement (m)
        '''
        record = self.get_eqsig_record()
        record.generate_displacement_and_velocity_series()
        return record.time, record.values, record.velocity, record.displacement


    def get_spectrum(self, periods=None):
        if "precomp_spectrum" in self.__dict__.keys():
            return self.precomp_spectrum
        else:
            spec = self.generate_response_spectrum(periods)
            # save spectrum for next run
            self.precomp_spectrum = spec
            if "database" in self.__dict__.keys():
                if self.database == "ngawest2":
                    folder = os.path.dirname(self.path).replace("\\time_histories", "\\spectra")
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    save_pickle(spec, os.path.join(folder, 
                                      os.path.basename(self.path).replace(".AT2","")))
            return spec


    def get_sa(self, period):
        if period == 0.:
            return self.pga
        else:
            if "interp_spectra" not in self.__dict__.keys():
                self.interp_spectra = self.get_interp_spectra()
            return float(self.interp_spectra(period))


    def get_avgsa_T(self, struct_period):
        '''
        The avgSa, selected here as IM, is conventionally calculated by
        considering a range of 10 equally-spaced periods spanning 
        approximately from a lower bound of 0.2T1 and an upper bound of 1.5T1,
        where T1 is the fundamental period of the structure (e.g., Kohrangi et 
        al. 2016).
        '''
        lower_period = 0.2*struct_period
        upper_period = 1.5*struct_period
        return self.get_avgsa(lower_period, upper_period)


    def get_avgsa(self, lower_period, upper_period):
        periods = np.linspace(lower_period, upper_period, 10)
        if "interp_spectra" not in self.__dict__.keys():
            self.interp_spectra = self.get_interp_spectra()
        return gmean( self.interp_spectra(periods) )


    def get_interp_spectra(self):
        '''
        spectra is a 2d nump array (n,2), with n number of response periods
        '''
        spectrum = self.get_spectrum()
        f = interpolate.interp1d(spectrum[:,0], spectrum[:,1])
        return f


    def plot_inputs(self, unit="m/s2", ax=None):
        '''
        quick and dirty plot to check
        '''
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlabel("Time (s)")
        if unit == "g":
            ax.plot(self.time, self.gmr/9.81)
            ax.set_ylabel("Acceleration (g)")
        elif unit == "m/s2":
            ax.plot(self.time, self.gmr)
            ax.set_ylabel("Acceleration (m/s2)")
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")
        if ax is None:
            plt.show()
        return ax


    def plot_arias_intensity(self, start=None, end=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.time, self.arias_intensity/self.arias_intensity[-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Arias intensity")
        if start is not None and end is not None:
            t1, t2 = self.get_significant_duration(start, end, se=True)
            ax.hlines(start, 0., t2, ls="--", lw=1., color="gray")
            ax.vlines(t1, 0., 1., ls="--", lw=1., color="gray")
            ax.hlines(end, 0., t2, ls="--", lw=1., color="gray")
            ax.vlines(t2, 0., 1., ls="--", lw=1., color="gray",
                      label="$D_{"+str(start*100)+"-"+str(end*100)+"}$")
            ax.legend(loc='lower right')
        if ax is None:
            plt.show()
        return ax


    def plot_husid(self, unit="m/s2", start=None, end=None, axs=None):
        if axs is None:
            fig, axs = plt.subplots(2, sharex=True)
        self.plot_inputs(unit, axs[0])
        self.plot_arias_intensity(start, end, axs[1])
        if axs is None:
            plt.show()
        return axs



    def plot_response_spectrum(self, periods=None, ax=None):
        period_sa = self.get_spectrum(periods)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(period_sa[:,0], period_sa[:,1])
        ax.set_xlabel("Response period (s)")
        ax.set_ylabel("Acceleration (m/s2)")
        if ax is None:
            plt.show()
        return ax


    def plot_comparison_spectra(self, ax=None):
        if "precomp_spectrum" not in self.__dict__.keys():
            raise Exception("no precomputed response spectrum was given")
        periods = self.precomp_spectrum[:,0]
        period_sa = self.generate_response_spectrum(periods)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(period_sa[:,0], period_sa[:,1])
        ax.plot(self.precomp_spectrum[:,0], self.precomp_spectrum[:,1])
        ax.set_xlabel("Response period (s)")
        ax.set_ylabel("Acceleration (m/s2)")
        if ax is None:
            plt.show()
        return ax


    def plot_acc_vel_disp(self, unit="m/s2", axs=None):
        time, acc, vel, disp = self.generate_acc_vel_disp_series()
        if axs is None:
            fig, axs = plt.subplots(3, sharex=True)
        if unit == "g":
            axs[0].plot(time, acc/9.81)
            axs[0].set_ylabel("Acceleration (g)")
        elif unit == "m/s2":
            axs[0].plot(time, acc)
            axs[0].set_ylabel("Acceleration (m/s2)")
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")
        axs[1].plot(time, vel)
        axs[1].set_ylabel("Velocity (m/s)")
        axs[2].plot(time, disp)
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Displacement (m)")
        if axs is None:
            plt.show()
        return axs


    @classmethod
    def load_simbad_record(cls, path, allinfo=False, spectrum_path=None,
                           metadata=None):
        data = loadmat(path)["record"]
        gmr = data[2:]
        t_step = data[1][0]
        kwargs = {"path": path, "database": "simbad"}
        if allinfo:
            # precomputed spectrum
            if spectrum_path is not None:
                precomp_spectrum = loadmat(spectrum_path)["spettro"]
                kwargs["precomp_spectrum"] = precomp_spectrum
            # metadata
            if metadata is None:
                two_up = os.path.abspath(os.path.join(path ,"../.."))
                metadata = pd.read_excel(os.path.join(two_up, "SIMBAD.xls"))
            filename = os.path.basename(path).replace(".mat","")
            meta = metadata.iloc[int(filename[2:6])-1]
            kwargs["rsn"] = meta["Waveform ID"]
            kwargs["eq_id"] = meta["Earthquake ID"]
            kwargs["eq_name"] = meta["Earthquake Name"]
            kwargs["date"] = meta["Date"]
            kwargs["time"] = meta["Time (UTC)"]
            kwargs["magnitude"] = meta["Mw"]
            kwargs["fault_mechanism"] = meta["Fault Mechanism"]
            kwargs["vs30"] = meta["Vs30 (m/s)"]
            kwargs["epicentral_distance"] = meta["Epicentral Distance (km)"]
            kwargs["site_class"] = meta["Site Class"].replace("*", "")
        self = cls(gmr, t_step=t_step, unit="m/s2", **kwargs)
        return self

    
    @classmethod
    def load_ngawest2_record(cls, path, allinfo=False, metadata=None):
        with open(path) as f:
            lines = f.readlines()
        t_step = float(lines[3].split(",")[1].split("=")[1].replace("\n","").replace("SEC","").replace("SE",""))
        n_pts = int(lines[3].split(",")[0].split("=")[1].replace("\n",""))
        gmr = list()
        for line in lines[4:]:
            for el in line.replace("\n","").split(" "):
                if el == "" or el.isspace():
                    continue
                gmr.append(float(el))
        if len(gmr) != n_pts:
            raise Exception("error in file", path)
        gmr = np.array(gmr)
        kwargs = {"path": path, "filename": os.path.basename(path), "database": "ngawest2"}
        if allinfo:
            # precomputed spectrum
            try:
                precomp_spectrum = load_pickle(path.replace(".AT2","").replace("\\time_histories\\", "\\spectra\\"))
                kwargs["precomp_spectrum"] = precomp_spectrum
            except:
                pass
            # metadata
            if metadata is None:
                two_up = os.path.abspath(os.path.join(path ,"../.."))
                metadata = cls.load_ngawest2_flatfile(two_up)
            filename = os.path.basename(path)
            recordname = filename.replace(".AT2", "")
            rsn = int(recordname.split("_")[0].replace("RSN", ""))
    
            meta = metadata.iloc[rsn-1]
            kwargs["rsn"] = meta["Record Sequence Number"]
            kwargs["eq_id"] = meta["EQID"]
            kwargs["eq_name"] = meta["Earthquake Name"]
            kwargs["date"] = str(meta["YEAR"])+" "+str(meta["MODY"])
            kwargs["time"] = meta["HRMN"]
            kwargs["magnitude"] = meta["Earthquake Magnitude"]
            kwargs["fault_mechanism"] = meta["Mechanism Based on Rake Angle"]
            kwargs["vs30"] = meta["Vs30 (m/s) selected for analysis"]
            kwargs["epicentral_distance"] = meta["EpiD (km)"]
            kwargs["jb_distance"] = meta["Joyner-Boore Dist. (km)"]
            kwargs["site_class"] = meta["Preferred NEHRP Based on Vs30"]
        self = cls(gmr, t_step=t_step, unit="g", **kwargs)
        return self

    
    @classmethod
    def load_ngawest2_flatfile(cls, path):
        flatfile = os.path.join(path,
                  "Updated_NGA_West2_Flatfile_RotD50_d050_public_version.xlsx")
        metadata = pd.read_excel(flatfile)
        return metadata


    @classmethod
    def load_goda_record(cls, path, allinfo=False, metadata=None):
        with open(path) as f:
            lines = f.readlines()
        gmr = list()
        for line in lines:
            gmr.append(float(line))
        gmr = np.array(gmr)
        other_path = os.path.join(os.path.dirname(path), "MSASinfo.mat")
        if metadata is None:
            metadata = loadmat(other_path)
        filename = os.path.basename(path).replace(".dat","")
        msas = filename.split("_")[0]
        h1h2 = filename.split("_")[1]
        _id = int(filename.split("_")[2])-1
        t_step = metadata["DT"][_id][0]
        kwargs = {"path": path, "database": "goda"}
        if allinfo:
            if other_path is not None:
                precomp_spectrum = np.vstack([metadata["Tn_RS"][0],
                                              metadata["PSA_Rec"+h1h2][_id]]).T
                kwargs["precomp_spectrum"] = precomp_spectrum
                
                kwargs["rsn"] = _id
                kwargs["eq_id"] = metadata["SequenceRecInfoMS"][_id,2]
                kwargs["eq_name"] = None
                kwargs["date"] = None
                kwargs["time"] = None
                kwargs["magnitude"] = metadata["SequenceRecInfoMS"][_id,3]
                kwargs["fault_mechanism"] = None
                kwargs["vs30"] = metadata["SequenceRecInfoMS"][_id,7]
                kwargs["epicentral_distance"] = None
                kwargs["jb_distance"] = None
                kwargs["rup_distance"] = metadata["SequenceRecInfoMS"][_id,6]
                kwargs["site_class"] = None
        '''
        SequenceRecInfoMS and SequenceRecInfoAS
        Column 1 : Database ID - 1) Crustal-NGA, 2) Crustal-KKiKSK, 3) Interface, 4) Inslab
        Column 2 : Record ID (based on original databases)
        Column 3 : Event ID (based on original databases)
        Column 4 : Magnitude
        Column 5 : Depth (km)
        Column 6 : Distance - Rhypo (km)
        Column 7 : Distance - Rrup (km)
        Column 8 : Vs30 (m/s)
        Column 9 : Network ID - 0) NGA database; 1) K-NET, 2) KiK-net (surface), 3) KiK-net (borehole), 4) SK-net
        Column 10: Number of records in a sequence (SequenceRecInfoMS only)
        '''
        self = cls(gmr, t_step=t_step, unit="m/s2", **kwargs)
        return self


    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        string = "<{} ".format(self.__class__.__name__) + \
               str(self.num_points) + " points" + \
               ", t_step: {:.4f}".format(self.t_step) + \
               ", pga: {:.3f}".format(self.pga) + "m/s2"
        if "database" in self.__dict__.keys():
            string = string + ", database: " + self.database
        return string+">"

