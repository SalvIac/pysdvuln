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

import numpy as np
import warnings
try:
    import openseespy.opensees as ops
except:
    warnings.warn("openseespy not installed!")


def get_sinewave(end_time=10., dt=1e-1, theta=0., frequency=1., amplitude=1.):
    '''
    end_time [s]
    df
    theta
    frequency = 1 # 1 cycle per second
    amplitude = 1 # in can be a numpy array
    '''
    start_time = 0. #s
    time = np.arange(start_time, end_time, dt)
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + 0.)
    return time, sinewave


class BaseSDOF():
    
    FREE = 0
    FIXED = 1
    
    X = 1
    Y = 2
    ROTZ = 3
    
    BOT_NODE = 1
    TOP_NODE = 2
    
    MAT_TAG = 1
    
    # 1kg (forces in N, work in Nm)
    # one could interpret this as 1ton (forces in kN, work in kNm)
    MASS = 1. # 1e3 
    
    G = 9.81
    
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


    def geometry(self):
        ops.model('basic', '-ndm', 1, '-ndf', 1)  # 1 dimension, 1 dof per node
    
        # establish nodes
        ops.node(self.BOT_NODE, 0.)
        ops.node(self.TOP_NODE, 0., '-mass', self.MASS) # nodal mass (force-weight / g)
    
        # fix bottom node
        ops.fix(self.BOT_NODE, self.FIXED)
    
        # assign zero length element
        beam_tag = 1
        ops.element('zeroLength', beam_tag, self.BOT_NODE, self.TOP_NODE,
                   "-mat", self.MAT_TAG, "-dir", 1, '-doRayleigh', 1)
    
    
    def material_martins_silva(self, capacity_curve, degradation=True,
                               collapse_factor=1.):
        '''
        capacity_curve is a pandas dataframe col1 Sd col2 Sa
        '''
        # if any(cap_df.duplicated()):
        #       warnings.warn("Warning: Duplicated pairs have been found in capacity curve!")
        
        d_cap = capacity_curve[:,0]
        f_cap = capacity_curve[:,1]*self.G*self.MASS # see MASS for units
        
        f_vec=np.zeros([5,1])
        d_vec=np.zeros([5,1])
        if len(f_cap)==3:
            #bilinear curve
            f_vec[1]=f_cap[1]
            f_vec[4]=f_cap[-1]
            
            d_vec[1]=d_cap[1]
            d_vec[4]=d_cap[-1]
            
            d_vec[2]=d_vec[1]+(d_vec[4]-d_vec[1])/3
            d_vec[3]=d_vec[1]+2*((d_vec[4]-d_vec[1])/3)
            
            f_vec[2]=np.interp(d_vec[2],d_cap,f_cap)
            f_vec[3]=np.interp(d_vec[3],d_cap,f_cap)
            
        elif len(f_cap)==4:
            # trilinear         
            f_vec[1]=f_cap[1]
            f_vec[4]=f_cap[-1]
            
            d_vec[1]=d_cap[1]
            d_vec[4]=d_cap[-1]
            
            f_vec[2]=f_cap[2]
            d_vec[2]=d_cap[2]
            
            d_vec[3]=np.mean([d_vec[2],d_vec[-1]])
            f_vec[3]=np.interp(d_vec[3],d_cap,f_cap)
            
        elif len(f_cap)==5:
            # trilinear with capacity loss
            f_vec[1]=f_cap[1]
            f_vec[4]=f_cap[-1]
            
            d_vec[1]=d_cap[1]
            d_vec[4]=d_cap[-1]
            
            f_vec[2]=f_cap[2]
            d_vec[2]=d_cap[2]
            
            f_vec[3]=f_cap[3]
            d_vec[3]=d_cap[3]
          
        matTag_pinching = 10
        if degradation == True:
            matargs = [f_vec[1,0], d_vec[1,0], f_vec[2,0], d_vec[2,0],
                       f_vec[3,0], d_vec[3,0], f_vec[4,0], d_vec[4,0],
                       -1*f_vec[1,0], -1*d_vec[1,0], -1*f_vec[2,0], -1*d_vec[2,0],
                       -1*f_vec[3,0], -1*d_vec[3,0], -1*f_vec[4,0], -1*d_vec[4,0],
                       0.5,0.25,0.05,
                       0.5,0.25,0.05,
                       0,0.1,0,0,0.2,
                       0,0.1,0,0,0.2,
                       0,0.4,0,0.4,0.9,
                       10,'energy']
        else:
            matargs = [f_vec[1,0], d_vec[1,0], f_vec[2,0], d_vec[2,0],
                       f_vec[3,0], d_vec[3,0], f_vec[4,0], d_vec[4,0],
                       -1*f_vec[1,0], -1*d_vec[1,0], -1*f_vec[2,0], -1*d_vec[2,0],
                       -1*f_vec[3,0], -1*d_vec[3,0], -1*f_vec[4,0], -1*d_vec[4,0],
                       0.5,0.25,0.05,
                       0.5,0.25,0.05,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       10,'energy']
    
        ops.uniaxialMaterial('Pinching4', matTag_pinching, *matargs)
        ops.uniaxialMaterial('MinMax', self.MAT_TAG, matTag_pinching,
                              '-min', -collapse_factor*d_vec[4,0],
                              '-max', collapse_factor*d_vec[4,0])
        return
    
    
    def material_hysteretic(self, capacity_curve):
        '''
        capacity_curve is a numpy array col1 Sd col2 Sa
        '''
        d_vec = capacity_curve[:,0]
        f_vec = capacity_curve[:,1]*self.G*self.MASS
        # Takeda type hysteresis
        # https://www.researchgate.net/post/Can_anyone_help_me_in_modeling_by_Takeda_hysteretic_rule_in_OpenSEES
        pitchx = pitchy = 1.
        damage1 = damage2 = 0.
        beta = 0.7 # 0.7 or 0.8
        if capacity_curve.shape[0] == 3:
            ops.uniaxialMaterial('Hysteretic', self.MAT_TAG,
                                 *[f_vec[1], d_vec[1]],
                                 *[f_vec[2], d_vec[2]],
                                 *[-f_vec[1], -d_vec[1]],
                                 *[-f_vec[2], -d_vec[2]], 
                                 pitchx, pitchy, damage1, damage2, beta)
        elif capacity_curve.shape[0] == 4:
            ops.uniaxialMaterial('Hysteretic', self.MAT_TAG,
                                 *[f_vec[1], d_vec[1]],
                                 *[f_vec[2], d_vec[2]],
                                 *[f_vec[3], d_vec[3]],
                                 *[-f_vec[1], -d_vec[1]],
                                 *[-f_vec[2], -d_vec[2]], 
                                 *[-f_vec[3], -d_vec[3]],
                                 pitchx, pitchy, damage1, damage2, beta)
        else:
            raise Exception("hystetic material not used for more than 4 points")
        return
    
    
    def material(self, mat_props):
        ops.uniaxialMaterial("Steel01", self.MAT_TAG, *mat_props)
        return
    
    
    def material_simple(self, capacity_curve):
        d_vec = capacity_curve[:,0]
        f_vec = capacity_curve[:,1]*self.G*self.MASS
        f_yield = f_vec[1]
        k_spring = f_vec[1]/d_vec[1]
        r_post = (f_vec[2]-f_vec[1])/(d_vec[2]-d_vec[1]) / k_spring
        if r_post == 0.:
            r_post = 1e-12
        mat_props = [f_yield, k_spring, r_post]
        ops.uniaxialMaterial("Steel01", self.MAT_TAG, *mat_props)
        return
    
    
    def get_secant_period(self, capacity_curve):
        '''
        secant (i.e., cracking point) period
        '''
        kx = capacity_curve[1,1]*self.G*self.MASS / capacity_curve[1,0]
        omega = np.sqrt(kx/self.MASS)
        period = 2*np.pi/omega
        return period
    
    
    def get_yield_period(self, capacity_curve):
        '''
        fundamental (i.e., yielding point) period
        this works with the capacity curves from Martins and Silva (2020)
        '''
        capacity_curve2 = self.mask_cracking_point(capacity_curve)
        return self.get_secant_period(capacity_curve2)
    
    
    @classmethod
    def mask_cracking_point(cls, capacity_curve):
        if capacity_curve.shape[0] > 3:
            # mask cracking point
            mask = np.ones(len(capacity_curve), dtype=bool)
            mask[[1]] = False
            capacity_curve2 = capacity_curve[mask,:]
        else:
            capacity_curve2 = capacity_curve    
        return capacity_curve2


    @classmethod
    def get_max_point(cls, capacity_curve):
        return (capacity_curve[np.argmax(capacity_curve[:,1]),0], 
                np.max(capacity_curve[:,1]))
    
    
    @classmethod
    def get_yielding_point(cls, capacity_curve):
        capacity_curve2 = cls.mask_cracking_point(capacity_curve)
        return capacity_curve2[1,0], capacity_curve2[1,1]
    
    
    def pushover_analysis(self, max_disp, d_disp):
        '''
        max_disp: maximum displacement [m]
        d_dist:   displacement increment [m]
        '''
    
        # Define constant axial load
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(self.TOP_NODE, 0.)
    
        ops.wipeAnalysis()
        
        # Define analysis parameters
        ops.integrator('LoadControl', 0.0)
        ops.system('SparseGeneral', '-piv')
        # ops.test('NormUnbalance', 1e-9, 1000)
        ops.test('NormDispIncr', 1.0e-12, 1000)
        ops.numberer('RCM')
        ops.constraints('Transformation')
        # ops.algorithm('Newton')
        ops.algorithm('ModifiedNewton', '-initial')
        ops.analysis('Static')
    
        ops.analyze(1)
    
        # incremental loading
        ops.timeSeries('Linear', 2)
        ops.pattern('Plain', 2, self.X)
        ops.load(self.TOP_NODE, 1.)
        
        ops.integrator('DisplacementControl', self.TOP_NODE, self.X, d_disp, 1, d_disp, d_disp)
    
        # run the analysis
        
        # Set some parameters
        currentDisp = 0.0
        ok = 0
        
        ops.reactions()
        outputs = {
            "disp": [ops.nodeDisp(self.TOP_NODE, self.X)],
            "force": [-ops.nodeReaction(self.BOT_NODE, self.X)]
        }
        
        while ok == 0 and currentDisp < max_disp:
        
            ok = ops.analyze(1)
        
            # if the analysis fails try initial tangent iteration
            if ok != 0:
                print("modified newton failed")
                break
            # print "regular newton failed .. lets try an initail stiffness for this step"
            # test('NormDispIncr', 1.0e-12,  1000)
            # # algorithm('ModifiedNewton', '-initial')
            # ok = analyze(1)
            # if ok == 0:
            #     print "that worked .. back to regular newton"
        
            # test('NormDispIncr', 1.0e-12,  10)
            # algorithm('Newton')
        
            currentDisp = ops.nodeDisp(self.TOP_NODE, self.X)
            ops.reactions()
            outputs["disp"].append(currentDisp)
            outputs["force"].append(-ops.nodeReaction(self.BOT_NODE, self.X))  # Negative since diff node
        ops.wipe()
    
        for item in outputs:
            outputs[item] = np.array(outputs[item])
        return outputs
    
    
    def cyclic_loading_analysis(self, displ_loading):
    
        # Define constant axial load
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(self.TOP_NODE, 0.)
    
        ops.wipeAnalysis()
        
        # Define analysis parameters
        ops.integrator('LoadControl', 0.0)
        ops.system('SparseGeneral', '-piv')
        # ops.test('NormUnbalance', 1e-9, 1000)
        ops.test('NormDispIncr', 1.0e-12, 1000)
        ops.numberer('RCM')
        ops.constraints('Transformation')
        # ops.algorithm('Newton')
        ops.algorithm('ModifiedNewton', '-initial')
        ops.analysis('Static')
    
        ops.analyze(1)
    
        # incremental loading
        ops.timeSeries('Linear', 2)
        ops.pattern('Plain', 2, self.X)
        ops.load(self.TOP_NODE, 1.)
        
        # run the analysis
        outputs = {
            "disp": [],
            "force": []
        }
        
        ok = 0
        D0 = 0.0
        for Dstep in displ_loading:
            D1 = Dstep
            Dincr = D1-D0
            ops.integrator("DisplacementControl", self.TOP_NODE, self.X, Dincr)
            ops.analysis("Static")
            ok = ops.analyze(1)
            # ----------------------------------------------if convergence failure-------------------------
            D0 = D1 # move to next step
            # end Dstep
            if ok != 0:
                print("Analysis failed at {} step.".format(Dstep))
            else:
                currentDisp = ops.nodeDisp(self.TOP_NODE, self.X)
                ops.reactions()
                outputs["disp"].append(currentDisp)
                outputs["force"].append(-ops.nodeReaction(self.BOT_NODE, self.X))  # Negative since diff node
        ops.wipe()
    
        for item in outputs:
            outputs[item] = np.array(outputs[item])
        return outputs
    
    
    def time_history_analysis(self, ground_motion, damping): # motion, dt
        '''
        ground motion col1 time [s], col2 accel [m/s2]
        '''
        # Define the dynamic analysis
        load_tag_dynamic = 10
        pattern_tag_dynamic = 100
        
        # values = list(-1 * motion)  # should be negative
        # ops.timeSeries('Path', load_tag_dynamic, '-dt', dt, '-values', *values)
        # ops.pattern('UniformExcitation', pattern_tag_dynamic, self.X, '-accel', load_tag_dynamic)
        
        gmr_values = -ground_motion[:,1]
        gmr_times = ground_motion[:,0]
        ops.timeSeries('Path', load_tag_dynamic, 
                       '-values', *gmr_values, '-time', *gmr_times)
        ops.pattern('UniformExcitation', pattern_tag_dynamic, self.X, '-accel', load_tag_dynamic)
    
        
        # set damping
        angular_freq = ops.eigen('-fullGenLapack', 1)[0] ** 0.5
        # two alternatives
        # alpha_m = 0.0
        # beta_k = 2*damping/angular_freq
        # # or
        alpha_m = 2*damping*angular_freq
        beta_k = 0.
        beta_k_comm = 0.0
        beta_k_init = 0.0
        ops.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)
    
        # Run the dynamic analysis
        ops.wipeAnalysis()
    
        # ops.constraints('Transformation')
        ops.constraints('Plain')
        ops.algorithm('Newton')
        # ops.system('SparseGeneral')
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        tol = 1.0e-10
        iterations = 50
        # ops.test('EnergyIncr', tol, iterations, 0, 2)
        ops.test('NormDispIncr', tol, iterations)
        
        t_final = ground_motion[-1,0] # (len(values) - 1) * dt
        t_current = ops.getTime()
        ok = 0
        outputs = {
            "time": [t_current],
            "disp": [0.],
            "vel": [0.],
            "accel": [0.],
            # "rel_disp": [0.],
            # "rel_vel": [0.],
            # "rel_accel": [0.],
            "force": [0.]
        }
        
        time_factor = 10
        analysis_dt = np.round(np.min(np.diff(ground_motion[:,0]))/1e-12)*1e-12/time_factor
        
        while ok == 0 and t_current < t_final:
            ok = ops.analyze(time_factor, analysis_dt)
            if ok != 0:
                print("regular newton failed... try initail stiffness for this step")
                ops.test('NormDispIncr', tol, iterations*10)
                # ops.test('EnergyIncr', tol, iterations*10, 0, 2)
                ops.algorithm('ModifiedNewton', '-initial')
                ok = ops.analyze(time_factor, analysis_dt)
                if ok != 0:
                    print("reducing dt by 10")
                    ndt = analysis_dt/10
                    ok = ops.analyze(time_factor*10, ndt)
                if ok == 0:
                    print("that worked ... back to regular settings")
                    ops.test('NormDispIncr', tol, iterations)
                    # ops.test('EnergyIncr', tol, iterations, 0, 2)
            t_current = ops.getTime()
            outputs["time"].append(t_current)
            outputs["disp"].append(ops.nodeDisp(self.TOP_NODE, self.X))
            outputs["vel"].append(ops.nodeVel(self.TOP_NODE, self.X))
            outputs["accel"].append(ops.nodeAccel(self.TOP_NODE, self.X))
            # outputs["rel_disp"].append(ops.nodeDisp(self.TOP_NODE, self.X)-ops.nodeDisp(self.BOT_NODE, self.X))
            # outputs["rel_vel"].append(ops.nodeVel(self.TOP_NODE, self.X)-ops.nodeVel(self.BOT_NODE, self.X))
            # outputs["rel_accel"].append(ops.nodeAccel(self.TOP_NODE, self.X)-ops.nodeAccel(self.BOT_NODE, self.X))
            ops.reactions()
            outputs["force"].append(-ops.nodeReaction(self.BOT_NODE, self.X)) # Negative since diff node
    
        # while ok == 0 and ops.getTime() < analysis_time:
        #     curr_time = ops.getTime()
        #     ops.analyze(1, analysis_dt)
        #     outputs["time"].append(curr_time)
        #     outputs["rel_disp"].append(ops.nodeDisp(self.TOP_NODE, 1))
        #     outputs["rel_vel"].append(ops.nodeVel(self.TOP_NODE, 1))
        #     outputs["rel_accel"].append(ops.nodeAccel(self.TOP_NODE, 1))
        #     ops.reactions()
        #     outputs["force"].append(-ops.nodeReaction(self.BOT_NODE, 1))  # Negative since diff node
        # ops.wipe()
        for item in outputs:
            outputs[item] = np.array(outputs[item])
      
        if ok!=0:
            print('NLTHA did not converge!')
                  
        return outputs
    
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "<{}, mass={}, G={}>".format(
                self.__class__.__name__, self.MASS, self.G)

