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
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
try:
    import openseespy.opensees as ops
except:
    import warnings
    warnings.warn("install openseespy (python >= 3.8)!")
from pysdvuln.base_sdof import BaseSDOF, get_sinewave
from matplotlib.animation import FuncAnimation
bsd = BaseSDOF()


class OpenseesRunner():
    
    DAMPING = 0.05
    DEGRADATION = True
    # Martins and Silva (2020) collapse is considered if the displacement
    # reaches 1.5 times the last displacement of the capacity curve
    COLLAPSE_FACTOR = 1.


    def __init__(self, capacity_curve, mass=1., **kwargs):
        if not np.all(capacity_curve[0,:] == 0.):
            capacity_curve = np.vstack([np.array([0.,0.]), capacity_curve])
        self.capacity_curve = capacity_curve
        self.__dict__.update(kwargs)
        self.mass = mass
        bsd.MASS = mass
    

    @classmethod
    def from_gem(cls, path, class_name):
        '''
        class_name: string
        '''
        filepath = os.path.join(path, class_name+".csv")
        capacity_curve = pd.read_csv(filepath).to_numpy()
        return cls(capacity_curve, class_name=class_name, filepath=filepath)


    def define_material(self, mat):
        if mat == "Pinching4":
            out = bsd.material_martins_silva(self.capacity_curve,
                                          degradation=self.DEGRADATION,
                                          collapse_factor=self.COLLAPSE_FACTOR)
        elif mat == "Hysteretic":
            out = bsd.material_hysteretic(self.capacity_curve)
        elif mat == "Steel01":
            out = bsd.material_simple(self.capacity_curve)
        else:
            raise Exception("unsupported material type!")
        return out
    

    def run(self, ground_motion, mat="Pinching4"):
        '''
        ground_motion: 2d numpy array
        '''
        ops.wipe()
        _ = self.define_material(mat)
        bsd.geometry()
        output = bsd.time_history_analysis(ground_motion, self.DAMPING)
        self.output = output
        # self.ground_motion = ground_motion
        return output
    
    
    def run_cyclic(self, mat="Pinching4", mode="increasing"):
        ops.wipe()
        _ = self.define_material(mat)
        bsd.geometry()
        if mode == "constant":
            time, sinewave = get_sinewave(amplitude=self.capacity_curve[-1,0]*0.9, dt=1e-2)
        elif mode == "increasing":
            ampls = np.linspace(0, self.capacity_curve[-1,0]*2., 1000)
            time, sinewave = get_sinewave(amplitude=ampls, dt=1e-2)
        else:
            raise Exception("check mode")
        output = bsd.cyclic_loading_analysis(sinewave)
        self.output = output
        self.output["time"] = time
        return output

    
    def calculate_hyst_energy(self, start=None, end=None):
        if start is None:
            start = self.output["time"][0]-0.1
        if end is None:
            end = self.output["time"][-1]+0.1
        inds = np.logical_and(self.output["time"] >= start, self.output["time"] < end) 
        return integrate.trapezoid(self.output["force"][inds]*self.output["vel"][inds],
                                   self.output["time"][inds])


    def get_max_disp(self, start=None, end=None, baseline=0.):
        if start is None:
            start = self.output["time"][0]-0.1
        if end is None:
            end = self.output["time"][-1]+0.1
        inds = np.logical_and(self.output["time"] >= start, self.output["time"] < end) 
        return np.max(np.abs(self.output["disp"][inds] - baseline))


    def get_value(self, key, time):
        if time > self.output["time"][-1]:
            raise Exception("input time > last time available!")
        ind = (np.abs(self.output["time"] - time)).argmin() # find nearest
        return self.output[key][ind]


    def get_disp(self, time):
        return self.get_value("disp", time)


    def get_yield_period(self):
        '''
        mass [kg]
        capacity curve [g]
        displacement [m]
        '''
        return bsd.get_yield_period(self.capacity_curve)


    def get_secant_period(self):
        '''
        mass [kg]
        capacity curve [g]
        displacement [m]
        '''
        return bsd.get_secant_period(self.capacity_curve)


    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}, capacity curve:\n{}>".format(
                self.__class__.__name__, self.capacity_curve)


    def plot_capacity_curve(self, ax=None, name=None, save=False, path=None):
        if ax is None:
            fig, ax = plt.subplots()
        if name is not None:
            ax.plot(self.capacity_curve[:,0], self.capacity_curve[:,1],
                    color="r", label=name)
        else:
            ax.plot(self.capacity_curve[:,0], self.capacity_curve[:,1],
                    color="r")
        ax.set_xlabel("Sd (m)")
        ax.set_ylabel("Sa (g)")
        if name is not None:
            ax.legend(framealpha=0.5)
        if save:
            if path is None:
                path = "capacity_curves.png"
            plt.savefig(path, bbox_inches='tight', dpi=600, format="png")
        else:
            plt.show()


    def plot_acc_disp_th(self, unit="m/s2"):
        fig, axs = plt.subplots(nrows=2, sharex=True)
        self.plot_disp_th(axs[0])
        self.plot_acc_th(unit, axs[1])
        plt.show()
        return axs
    
    
    def plot_disp_th(self, ax=None, start_time=None, end_time=None):
        if ax is None:
            fig, ax = plt.subplots()
        ind = np.array([True]*len(self.output["time"]))
        if start_time is not None:
            ind = np.logical_and(ind, self.output["time"] >= start_time)
        if end_time is not None:
            ind = np.logical_and(ind, self.output["time"] < end_time)
        ax.plot(self.output["time"][ind], self.output["disp"][ind], lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        if ax is None:
            plt.show()
        return ax

            
    def plot_acc_th(self, unit="m/s2", ax=None, start_time=None, end_time=None):
        if ax is None:
            fig, ax = plt.subplots()
        ind = np.array([True]*len(self.output["time"]))
        if start_time is not None:
            ind = np.logical_and(ind, self.output["time"] >= start_time)
        if end_time is not None:
            ind = np.logical_and(ind, self.output["time"] < end_time)
        if unit == "g":
            ax.plot(self.output["time"][ind], self.output["accel"][ind]/9.81, lw=1)
        else:
            ax.plot(self.output["time"][ind], self.output["accel"][ind], lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration ({})".format(unit))
        if ax is None:
            plt.show()
        return ax
    

    def plot_force_disp_wth(self, ax2="acc", unit="m/s2", start_time=None,
                            end_time=None):
        fig, axs = plt.subplots(2,1)
        self.plot_force_disp(axs[0], start_time, end_time)
        if ax2 == "acc":
            self.plot_acc_th(unit, axs[1], start_time, end_time)
        elif ax2 == "disp":
            self.plot_disp_th(axs[1], start_time, end_time)
        else:
            raise Exception("check ax2")
        plt.show()
        return axs


    def plot_force_disp(self, ax=None, start_time=None, end_time=None):
        if ax is None:
            fig, ax = plt.subplots()
        ind = np.array([True]*len(self.output["time"]))
        if start_time is not None:
            ind = np.logical_and(ind, self.output["time"] >= start_time)
        if end_time is not None:
            ind = np.logical_and(ind, self.output["time"] < end_time)
        ax.plot(self.output["disp"][ind], self.output["force"][ind], lw=1)
        ax.scatter(self.output["disp"][ind][-1], self.output["force"][ind][-1],
                   facecolor="k", edgecolor="k")
        ax.plot(self.capacity_curve[:,0], self.capacity_curve[:,1]*9.81*bsd.MASS,
                lw=1, color="k", ls="--")
        ax.plot(-self.capacity_curve[:,0], -self.capacity_curve[:,1]*9.81*bsd.MASS,
                lw=1, color="k", ls="--")
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Base shear (N per unit mass)")
        if ax is None:
            plt.show()
        return ax
    
    
    def animate(self):
        raise Exception("incomplete")
        # animation
        fig, self.axs = plt.subplots(2,1, figsize=(8,12))
        if np.max(self.output["disp"]) > np.max(self.capacity_curve[:,0]):
            self.axs[0].set_xlim(np.min(self.output["disp"])*1.1,
                            np.max(self.output["disp"])*1.1)
        else:
            self.axs[0].set_xlim(-np.max(self.capacity_curve[:,0])*1.1,
                            np.max(self.capacity_curve[:,0])*1.1)
        self.axs[0].set_ylim(-np.max(self.capacity_curve[:,1])*9.81*bsd.MASS*1.1, 
                        np.max(self.capacity_curve[:,1])*9.81*bsd.MASS*1.1)
        self.axs[0].set_xlabel("Displacement (m)")
        self.axs[0].set_ylabel("Base shear (N)")
        self.axs[1].set_xlim(np.min(self.output["time"]), np.max(self.output["time"]))
        self.axs[1].set_ylim(np.min(self.output["accel"])/9.81*1.1, 
                        np.max(self.output["accel"])/9.81*1.1)
        self.axs[1].set_xlabel("Time (s)")
        self.axs[1].set_ylabel("Acceleration (g)")

        self.axs[0].plot(self.output["disp"], self.output["force"],
                    color=[0.7,0.7,0.7], lw=1)
        self.axs[1].plot(self.output["time"], self.output["accel"]/9.81,
                    color=[0.7,0.7,0.7], lw=1)
        self.axs[0].plot(self.capacity_curve[:,0], self.capacity_curve[:,1]*9.81*bsd.MASS,
                    lw=1, color="k", ls="--")
        self.axs[0].plot(-self.capacity_curve[:,0], -self.capacity_curve[:,1]*9.81*bsd.MASS,
                    lw=1, color="k", ls="--")
        self.anim = FuncAnimation(fig, self.animation_step,# init_func=init,
                                  frames=self.output["time"].shape[0],
                                  interval=5)#self.output["time"][1]-self.output["time"][0])

        
    def animation_step(self, i):
        self.axs[0].plot(self.output["disp"][:i], self.output["force"][:i],
                    color="b", lw=1) #, marker="o", markersize=2)
        self.axs[1].plot(self.output["time"][:i], self.output["accel"][:i]/9.81,
                    color="b", lw=1) #, marker="o", markersize=2)
        

