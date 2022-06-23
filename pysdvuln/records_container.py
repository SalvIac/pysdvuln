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
from glob import glob
import warnings
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat
# from scipy.stats import qmc
import matplotlib.pyplot as plt
from pysdvuln.record import Record
from pysdvuln.record_two_comp import RecordTwoComp, RecordChooseComp


class RecordsContainer():
    
    def __init__(self, records=[], database=None, sourcepath=None, seed=1992):
        self.records = records
        self.sourcepath = sourcepath
        self.database = database
        self.excluded = list()
        self.ims = dict()
        self.spectra = dict()
        self.interp_spectra = dict()
        np.random.seed(seed)
        # if "ngawest2" in databases: #TODO here I could use a callable dict
        #     self.load_ngawest2_records()
        # if "simbad" in databases:
        #     self.load_simbad_records()
        # if "goda" in databases:
        #     self.load_goda_records()
        
    
    @classmethod
    def load_simbad_data(cls, sourcepath):
        path_records = os.path.join(sourcepath, "SIMBAD (Updated 2011)", 
                                    "Records")
        path_spectra = os.path.join(sourcepath, "SIMBAD (Updated 2011)",
                                    "Spectra")
        all_record_paths = glob(os.path.join(path_records,"*.mat"))
        flatfile = os.path.join(sourcepath, "SIMBAD (Updated 2011)",
                                "SIMBAD.xls")
        metadata = pd.read_excel(flatfile)
        records = list()
        for path in tqdm(all_record_paths):
            filename = os.path.basename(path)
            if "za_" not in filename:
                spectrum_name = filename.replace("_record", "")
                spectrum_path = os.path.join(path_spectra, spectrum_name)
                rec = Record.load_simbad_record(path, allinfo=True,
                                                spectrum_path=spectrum_path,
                                                metadata=metadata)
                records.append(rec)
        return cls(records, database="simbad", sourcepath=sourcepath)
    
    
    @classmethod
    def load_ngawest2_data(cls, sourcepath):
        path_records = os.path.join(sourcepath, "NGAWest2", "time_histories")
        all_record_paths = glob(os.path.join(path_records,"*.AT2"))
        metadata = Record.load_ngawest2_flatfile(os.path.join(sourcepath, "NGAWest2"))
        vertical = list()
        for st, rsn in zip(metadata["File Name (Vertical)"], metadata["Record Sequence Number"]):
            if st == -999 or st == "-999":
                continue
            if "\\" in st:
                sym = "\\"
            else:
                sym = "/"
            vertical.append( "RSN"+str(rsn)+"_"+st.replace(sym, "_") )
        records = list()
        for path in tqdm(all_record_paths):
            filename = os.path.basename(path)
            rsn = int(filename.split("_")[0].replace("RSN", ""))
            if "-UP." in filename or "V." in filename or "V1." in filename or \
                "DWN." in filename or "DN." in filename or "UD." in filename or \
                "Z." in filename:
                continue
            if filename not in vertical:
                rec = Record.load_ngawest2_record(path, allinfo=True,
                                                  metadata=metadata)
                records.append(rec)
        return cls(records, database="ngawest2", sourcepath=sourcepath)


    @classmethod
    def load_ngawest2_data_comps(cls, sourcepath):
        path_records = os.path.join(sourcepath, "NGAWest2", "time_histories")
        all_record_paths = glob(os.path.join(path_records,"*.AT2"))
        all_record_filenames = [os.path.basename(path) for path in all_record_paths]
        metadata = Record.load_ngawest2_flatfile(os.path.join(sourcepath, "NGAWest2"))
        records = list()
        for st1, st2, rsn in zip(tqdm(metadata["File Name (Horizontal 1)"]),
                                 metadata["File Name (Horizontal 2)"],
                                 metadata["Record Sequence Number"]):
            if st1 == -999 or st1 == "-999" or st2 == -999 or st2 == "-999":
                continue
            name1 = "RSN"+str(rsn)+"_"+st1.replace("\\", "_").replace("/", "_")
            name2 = "RSN"+str(rsn)+"_"+st2.replace("\\", "_").replace("/", "_")
            if name1 not in all_record_filenames or name2 not in all_record_filenames:
                continue
            rec1 = Record.load_ngawest2_record(os.path.join(path_records, name1),
                                               allinfo=True, metadata=metadata)
            rec2 = Record.load_ngawest2_record(os.path.join(path_records, name2),
                                               allinfo=True, metadata=metadata)
            trec = RecordTwoComp(rec1, rec2)
            records.append(trec)
        return cls(records, database="ngawest2", sourcepath=sourcepath)
    
    
    @classmethod
    def load_goda_data(cls, sourcepath):
        path_records = os.path.join(sourcepath, "MS-AS sequences by Katsu Goda")
        all_record_paths = glob(os.path.join(path_records,"*.dat"))
        other_path = os.path.join(path_records, "MSASinfo.mat")
        data = loadmat(other_path)
        records = list()
        for path in tqdm(all_record_paths):
            if os.path.basename(path).split("_")[0] == "MS":
                rec = Record.load_goda_record(path, allinfo=True, metadata=data)
                records.append(rec)
        return cls(records, database="goda", sourcepath=sourcepath)

    
    def inconsistent_t_steps(self):
        t_steps = list()
        for gmr in self.records:
            t_steps.append(gmr.t_step)
        min_t_step = min(t_steps)
        remain0 = [t_step % min_t_step != 0 for t_step in t_steps]
        if any(remain0):
            inds = np.where(remain0)[0][::-1]
            string = "\n".join([str(self.records[i]) for i in inds])
            warnings.warn("following ground-motion records excluded because of t_step:\n"+string)
            for i in inds:
                self.excluded.append(deepcopy(self.records[i]))
                del self.records[i]
    
    
    def check_duplicates(self):
        r0 = self.get_sas(0.)
        r0r = np.round(r0, 4)
        u, c = np.unique(r0r, return_counts=True)
        dup = u[c > 1]
        todelete = list()
        periods = [0.01, 0.05, 0.075, 0.1, 0.2, 0.5, 1., 2.]
        for d in tqdm(dup):
            inds = np.where(r0r == d)[0]
            sref = self.records[inds[0]].generate_response_spectrum(periods)
            for ind in inds[1:]:
                sche = self.records[ind].generate_response_spectrum(periods)
                # ax = self.records[inds[0]].plot_response_spectrum()
                # self.records[ind].plot_response_spectrum(ax=ax)
                if np.mean(np.power(sref-sche, 2)) < 1e-4:
                    todelete.append(ind)
        return todelete
    

    def delete_records(self, inds):
        if not hasattr(inds, '__iter__'):
            inds = [inds]
        inds = sorted(inds)
        for i in inds[::-1]:
            self.excluded.append(deepcopy(self.records[i]))
            for key in ["spectra", "interp_spectra"]:
                if self.records[i] in self.__dict__[key].keys():
                    del self.__dict__[key][self.records[i]]
            del self.records[i]
        for key in ["ims"]:
            for key2 in self.__dict__[key].keys():
                self.__dict__[key][key2] = np.delete(self.__dict__[key][key2], inds)

    
    @property
    def num_records(self):
        return len(self.records)


    def get_array(self, key):
        array = list()
        for rec in self.records:
            array.append( rec.__dict__[key] )
        return np.array(array)


    def get_sas(self, period):
        if period == 0.:
            key = "PGA"
        else:
            key = "SA({:.1f})".format(period)
        if key in self.ims.keys():
            return self.ims[key]
        ims = list()
        for rec in self.records:
            ims.append(rec.get_sa(period))
        self.ims[key] = np.array(ims)
        return self.ims[key]
    

    def get_avgsas_T(self, struct_period):
        key = "avgSA({:.2f})".format(struct_period)
        if key in self.ims.keys():
            return self.ims[key]
        ims = list()
        for rec in self.records:
            ims.append(rec.get_avgsa_T(struct_period))
        self.ims[key] = np.array(ims)
        return self.ims[key]

    
    def get_avgsas(self, lower_period, upper_period):
        key = "avgSA({:.2f},{:.2f})".format(lower_period, upper_period)
        if key in self.ims.keys():
            return self.ims[key]
        ims = list()
        for rec in self.records:
            ims.append(rec.get_avgsa(lower_period, upper_period))
        self.ims[key] = np.array(ims)
        return self.ims[key]
    
    
    def plot_all_spectra(self, unit="g", x_scale="linear"):
        fig, ax = plt.subplots()
        ax.set_xlabel("Period (s)")
        if unit == "g":
            ax.set_ylabel("Acceleration (g)")
        elif unit == "m/s2":
            ax.set_ylabel("Acceleration (m/s2)")
        else:
            raise Exception("unit can only be 'g' or 'm/s2'")
        for rec in self.records:
            spec = rec.get_spectrum()
            if unit == "g":
                spec[:,1] = spec[:,1]/9.81
            ax.plot(spec[:,0], spec[:,1], linewidth=0.5, color=[0.5,0.5,0.5])
        ax.set_xscale(x_scale)
        plt.show()
    
    
    #TODO the next four functions could be changed to callable dicts
    def get_durations(self):
        duration = list()
        for rec in self.records:
            duration.append( rec.duration )
        return np.array(duration)


    def get_sign_durations(self, start=0.05, end=0.95):
        sign_dur = list()
        for rec in self.records:
            sign_dur.append( rec.get_significant_duration(start, end) )
        return np.array(sign_dur)


    def filter_duration(self, cutoff=10):
        durs = self.get_durations()
        inds = np.where(durs < cutoff)[0]
        self.delete_records(inds.tolist())


    def filter_sign_duration(self, cutoff=5, start=0.05, end=0.95):
        durs = self.get_sign_durations(start, end)
        inds = np.where(durs < cutoff)[0]
        self.delete_records(inds.tolist())

   


    @classmethod
    def scatter(cls, array1, array2, xlabel="", ylabel="", **kwargs):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(array1, array2, **kwargs)#s=20, linewidth=0.1, edgecolors='k', color='gray')
        ax.set_xlabel("GM1 "+xlabel)
        ax.set_ylabel("GM2 "+ylabel)
        plt.show()
        
        
    @classmethod
    def hist2d(cls, array2d, bar_width=1., range_bin=None, centered=False,
               xlabel="", ylabel=""):
        if range_bin is not None:
            hist, xedges, yedges = np.histogram2d(array2d[:,0], array2d[:,1],
                                             bins=20, range=[range_bin, range_bin])
        else:
            hist, xedges, yedges = np.histogram2d(array2d[:,0], array2d[:,1], bins=20)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if centered:
            xpos, ypos = np.meshgrid(xedges[:-1] + np.min(np.diff(xedges))/2,
                                     yedges[:-1] + np.min(np.diff(yedges))/2, 
                                     indexing="ij")
        else:
            xpos, ypos = np.meshgrid(xedges[:-1], xedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        dx = dy = bar_width*np.ones_like(zpos)
        dz = hist.ravel()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
        ax.set_xlabel("GM1 "+xlabel)
        ax.set_ylabel("GM2 "+ylabel)
        plt.show()
    


    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{} ".format(self.__class__.__name__) + \
               str(self.num_records) + " records>"
   


    # def __iter__(self):
    #     self.__n = 0
    #     return self

    # def __next__(self):
    #     if self.__n < self.num_catalogs:
    #         ob = self[self.__n]
    #         self.__n += 1
    #         return ob
    #     else:
    #         raise StopIteration
    
    # def __getitem__(self, i):
    #     ob = self.gmf_catalogs[i]
    #     if isinstance(ob, list):
    #         for o in ob:
    #             if o.sites is None:
    #                 o.sites = self.sites
    #             if o.imts is None:
    #                 o.imts = self.imts
    #     else:
    #         if ob.sites is None:
    #             ob.sites = self.sites
    #         if ob.imts is None:
    #             ob.imts = self.imts
    #     return ob
