#%% NLTHA TELAIO PRE 70 #########################################################

import numpy as np
import pylab as pl
import subprocess
import os
import pickle
import sys
sys.path.insert(1, 'D:\Desktop\Function_Python')

import fun_Sect_Analysis_RMK as FunSecAn

#%% RISULTATI ###################################################################

runfile("1_INPUT_DATA.py")

f = open('Result_NLTH_SDoF_dict.pckl', 'rb')
Result_NLTH_SDoF_dict = pickle.load(f)
f.close()

for key,val in Result_NLTH_SDoF_dict.items():
        exec(key + '=val')

# SPECTRA
if 'Acc_MSAS' in globals():
    print('File records already runed')
else:
    f = open('Selected_sequences.pckl', 'rb')
    Selected_sequences = pickle.load(f)
    f.close()
    
    for key,val in Selected_sequences.items():
        exec(key + '=val') 


#%% UTILE #######################################################################

import psutil
    
def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False;


#%% Eq SDoF RUAUMOKO =======================================#

# PARAMETERS SDoF
f = open('Parameters_SDoF_dict.pckl', 'rb')
Parameters_SDoF_dict = pickle.load(f)
f.close()

for key,val in Parameters_SDoF_dict.items():
        exec(key + '=val')

# KILL WIN ERROR
subprocess.Popen([r'D:\Desktop\KillQuickWinErr.exe'])

#second_analyses = np.where(Drift_max_MSAS == 0)[0]

#for MSAS in range(len(Acc_MSAS)):
#for MSAS in [5,58,187,195,259,278,318,338,384,437,453,472,474,551,591]:
for MSAS in [2]:
#for MSAS in second_analyses:
    
    MSAS = int(MSAS)
    
    # INPUT PER AZIONE ------------------------------------------------------------
    DT = 0.001
    T_max = Time_MSAS[MSAS][-1] + 20
    
    print(Time_MS[MSAS][-1])
    print(T_max)
    
    STEP_DYNPL = 10

    #%% CANCELLAZIONE FILE ESISTENTI ################################################
    
    file_del = ['SDoF_TH.txt',
                'Model_RMK_BAT.BAT',
                'OUTPUT_'+str(MSAS)+'.WRI',
                'OUTPUT_'+str(MSAS)+'.RES',
                'LOG_DYNPL.txt',
                'DYNPL.WRI',
                'DYNPL_BAT.BAT',
                'WORK.txt',
                'DISPL.txt']
    
    for i in file_del:
        try:
            os.remove(i)
       
        except OSError:
            print('File non presente nella cartella: '+i)
            
    
            
    #%% MODELLO RUAUMOKO2D #############################################
    
    Model_RMK = open('SDoF_TH.txt','w')
    
    # BLOCCO 1 - PARAMETRI PER L'ANALIS E PER IL MODELLO ==========================
    
    # Linea 1 = Descrizione della struttura (fino a 79 caratteri alfanumerici)
    Model_RMK.write('SDoF_TH')
    
    # Linea 2 = Principal analysis option
    Model_RMK.write("\n"+"2 0 1 0 2 0 0 0 0 0 0")
    
    # Linea 3 = Frame control parameters    
    Model_RMK.write("\n"+"2 1 1 1 1 2 9.81 5.0 5.0 "+str(DT)+" "+str(T_max)+" 1")
    
    # Linea 4 = output intervals and plotting
    Model_RMK.write("\n"+"0 "+str(STEP_DYNPL)+" "+str(STEP_DYNPL)+" 0 1 5 1 1 "+str(1)+" 2 0 0")
    
    # Linea 5 = Iteration control
    Model_RMK.write("\n"+'100 10 '+str(10**(-8))+' 0 0 0 ')
    
    Model_RMK.write("\n")
    
    # BLOCCO 2 - GEOMETRIA E VINCOLI ==============================================
    
    # Nodi ------------------------------------------------------------------------
    Model_RMK.write("\n"+"NODES")
 
    #N X(N) Y(N) NF1 NF2 NF3 KUP1 KUP2 KUP3 IOUT
    Model_RMK.write("\n"+"1 0.0 0.0 1 1 1 0 0 0 0")
    Model_RMK.write("\n"+"2 1.0 0.0 0 1 1 0 0 0 0")
    
    Model_RMK.write("\n")
           
    # BLOCCO 3 - CONNETTIVITà ELEMENTI ============================================
    
    Model_RMK.write("\n"+"ELEMENTS 0")
    
    Model_RMK.write("\n"+" 1 1 1 2")
    
    Model_RMK.write("\n")    
    
            
    # BLOCCO 4 - PROPIETà ELEMENTI ================================================
    
    Model_RMK.write("\n"+"PROPS")  
    Model_RMK.write("\n"+"1 SPRING") 
    
    Model_RMK.write("\n"+"1 52 0 0 ")                        
    Model_RMK.write(str(K_0)+' ')          # KX
    Model_RMK.write(' 0')          # KY
    Model_RMK.write(' 0')          # GJ
                       
    Model_RMK.write(" 0.0 "+str(r)+" "+str(r))      # WGT, RF, RT
    
    Model_RMK.write("\n"+str(F_SDoF[1])+" "+str(-F_SDoF[1])+" 0 0 0 0")
    
    i = 0
    while i < 3:        
        Model_RMK.write("\n"+str(Kneg)+" ")
        Model_RMK.write(str(Rbeg)+" ") 
        Model_RMK.write(str(Fcr_pos)+" ") 
        Model_RMK.write(str(Fcr_neg)+" ")
        Model_RMK.write(str(Rho_pos)+" ")  
        Model_RMK.write(str(Rho_neg)+" ")  
        Model_RMK.write(str(Dult_pos)+" ")
        Model_RMK.write(str(Dult_neg)+" ") 
        Model_RMK.write(str(IOP)+" ")      
        
        Model_RMK.write("\n"+str(Alpha)+" ")
        Model_RMK.write(str(Beta)+" ")      
        Model_RMK.write(str(Pinch)+" ")     
        Model_RMK.write(str(Kappa_pos)+" ") 
    #    Model_RMK.write(str(Kappa_neg)+" ") 
        Model_RMK.write(str(Fresid)+" ")    
        Model_RMK.write(str(Dfactor)+" ") 
        
        i = i+1
    
    Model_RMK.write("\n")
                
    # BLOCCO 5 - MASSE DI PIANO ###################################################
       
    Model_RMK.write("\n"+"WEIGHTS 0")
    Model_RMK.write("\n"+"2 "+str(m_eff)+" 0 0")
    
    Model_RMK.write("\n")
    
    
    # BLOCCO 6 - CARICHI VERTICALI ################################################
    
    Model_RMK.write("\n"+"LOADS")
    
    Model_RMK.write("\n"+"1 0.0 0.0 0.0")
    Model_RMK.write("\n"+"2 0.0 0.0 0.0")
            
    Model_RMK.write("\n")    
    
    # BLOCCO 8 - INPUT PER SOLLECITAZIONI E PARAMETRI DI CONTROLLO ################
    Model_RMK.write("\n"+"EQUAKE")
    Model_RMK.write("\n"+"3 1 ")
    Model_RMK.write(str(Time_MSAS[MSAS][1] - Time_MSAS[MSAS][0]))
    Model_RMK.write(" 1 -1 0 0 1")
    
    Model_RMK.write("\n"+"START")
    
    NOME = 0
    
    for i in range(len(Time_MSAS[MSAS])):
        
        Model_RMK.write("\n"+str(NOME)+" ")
        Model_RMK.write(str(Time_MSAS[MSAS][i])+" ")
        Model_RMK.write(str(Acc_MSAS[MSAS][i]))
        
        NOME = NOME+1
        
    for i in range(int(20/(Time_MSAS[MSAS][1] - Time_MSAS[MSAS][0]))):
        Model_RMK.write("\n"+str(NOME)+" ")
        Model_RMK.write(str(Time_MSAS[MSAS][-1]+(i+1)*(Time_MSAS[MSAS][1] - Time_MSAS[MSAS][0]))+" 0")
        
        NOME = NOME+1
    
    Model_RMK.close()
    
    
    #%% FILE BATCH RUAUMOKO #########################################################
    
    path = os.getcwd()
    
    File_batch_RM = open('Model_RMK_BAT.BAT','w')
    File_batch_RM.write('@ECHO OFF')
    File_batch_RM.write("\n"+path)
    File_batch_RM.write('\RunAsDate_x64.exe 01/06/2020 12:35:22 ')
    File_batch_RM.write(path)
    File_batch_RM.write('\Ruaumoko2N_UCL.exe OUTPUT_'+str(MSAS)+'.WRI SDoF_TH.txt')
    File_batch_RM.write("\n"+'@ECHO ON')
    
    File_batch_RM.close() 
    
    # Run analysis
    subprocess.Popen([r'Model_RMK_BAT.BAT'])  
    
    import time
    time.sleep(5)
    
    import win32com.client as win32
    
    shell = win32.Dispatch("WScript.Shell")
    
    if shell.AppActivate("Intel(r) Visual Fortran run-time error") == True:
        shell.Sendkeys("{ENTER}", 0)

    
    # Check if Ruaumoko was running or not.
    while checkIfProcessRunning('Ruaumoko2N_UCL') == True:   
        
        shell = win32.Dispatch("WScript.Shell")
        
        if shell.AppActivate("Intel(r) Visual Fortran run-time error") == True:
            shell.Sendkeys("{ENTER}", 0)
        
        time.sleep(2)
    
#    shell.AppActivate("Kill QuickWinError Script")
#    shell.Sendkeys("{ENTER}", 0)    
    
    
    #%% FILE TXT DYNAPLOT ###########################################################
        
    # ENERGIES
    
    File_LOG_DYNPL= open('LOG_DYNPL.txt','w')
    File_LOG_DYNPL.write('DYNPL.WRI')
    File_LOG_DYNPL.write("\n"+"OUTPUT_"+str(MSAS)+" NO")
    File_LOG_DYNPL.write("\n"+"No")
    File_LOG_DYNPL.write("\n"+"W")
    File_LOG_DYNPL.write("\n"+" ")
    File_LOG_DYNPL.write("\n"+"K")
    File_LOG_DYNPL.write("\n"+"WORK.txt")
    File_LOG_DYNPL.write("\n"+"B")
    File_LOG_DYNPL.write("\n"+"S")
    
    File_LOG_DYNPL.close()
    
    
    #%% FILE BATCH DYNPLOT #########################################################
    
    File_batch_DNPL = open('DYNPL_BAT.BAT','w')
    File_batch_DNPL.write('Dynaplan DYNPL.wri OUTPUT_'+str(MSAS)+' LOG_DYNPL.txt')
    File_batch_DNPL.close()
    
    subprocess.call([r'DYNPL_BAT.BAT'])
    
    
        
    #%% INTERSTORY-DRIFT (NODAL DISPLACEMENTS)
    
    remove_DYNPL_file = ['LOG_DYNPL.txt',
                         'DYNPL.WRI',
                         'DYNPL_BAT.BAT',]
    
    for i in remove_DYNPL_file:
        os.remove(i)
    
    File_LOG_DYNPL= open('LOG_DYNPL.txt','w')
    File_LOG_DYNPL.write('DYNPL.WRI')
    File_LOG_DYNPL.write("\n"+"OUTPUT_"+str(MSAS)+" NO")
    File_LOG_DYNPL.write("\n"+"No")
    File_LOG_DYNPL.write("\n"+"T")
    File_LOG_DYNPL.write("\n"+"1")
    File_LOG_DYNPL.write("\n"+'N')
    File_LOG_DYNPL.write("\n"+'1')
    
    File_LOG_DYNPL.write("\n"+'2')
    
    File_LOG_DYNPL.write("\n"+"R")    
    File_LOG_DYNPL.write("\n"+"0.0")
    File_LOG_DYNPL.write("\n"+"-1 1")
    File_LOG_DYNPL.write("\n"+"1.0")
    File_LOG_DYNPL.write("\n"+" ")
    File_LOG_DYNPL.write("\n"+" ")
    File_LOG_DYNPL.write("\n"+" ")
    File_LOG_DYNPL.write("\n"+"K")
    File_LOG_DYNPL.write("\n"+"DISPL.txt")
    File_LOG_DYNPL.write("\n"+"B")
    File_LOG_DYNPL.write("\n"+"S")
    
    File_LOG_DYNPL.close()
    
    
    #%% FILE BATCH DYNPLOT ##########################################################
    
    File_batch_DNPL = open('DYNPL_BAT.BAT','w')
    File_batch_DNPL.write('Dynaplan DYNPL.wri OUTPUT_'+str(MSAS)+' LOG_DYNPL.txt')
    File_batch_DNPL.close()
    
    subprocess.call([r'DYNPL_BAT.BAT'])
    
    
    #%% ELABORAZIONE DATI ###########################################################
    
    step_final_MS = int(Time_MS[MSAS][-1]/DT/STEP_DYNPL)
    step_iniz_AS  = int((Time_MS[MSAS][-1]+20)/DT/STEP_DYNPL)
    
    # ENERGIES
    T, Kin, Kin_Damp, Kin_Damp_Strain, Ex_work  = np.loadtxt('WORK.txt', unpack = 'True')
        
    # SPOSTAMENTI  E DRIFT E SPOSTAMENTI RESIDUI
    Displ  = np.loadtxt('DISPL.txt')
        
    # CHECK COLLAPSE OR ERROR
    if Displ[-1][0] < T_max-10:
        print('Dynamic Instability')
        
    else:             
        E_HYST_MSi   = Kin_Damp_Strain[step_final_MS] - Kin_Damp[step_final_MS]
        E_HYST_MSASi = Kin_Damp_Strain[-1] - Kin_Damp[-1]
        E_HYST_ASi   = E_HYST_MSASi-E_HYST_MSi
        
        print("E_HYST_MS = ", E_HYST_MSi)
        print("E_HYST_AS = ", E_HYST_ASi)
        print("E_HYST_MSAS = ", E_HYST_MSASi)    

        s_roof   = Displ[:,1]/coeff_disp
        s_storey = np.zeros((len(s_roof), len(delta_i_adm)))
        
        for i in range(len(Displ[:,1])):
            s_storey[i] =  s_roof[i]*delta_i_adm
        
        Drift = np.zeros((len(s_storey[:,0]),len(s_storey[1])))
        
        Drift_max_storey_MS   = np.zeros(Npiani)
        Drift_max_storey_AS   = np.zeros(Npiani)    
        Drift_max_storey_MSAS = np.zeros(Npiani)
        
        for z in range(Npiani):
            if z == 0:
                Drift[:,z] = s_storey[:,z]/(Z[z+1]-Z[z])
            else:
                Drift[:,z] = (s_storey[:,z]-s_storey[:,z-1])/(Z[z+1]-Z[z])        
        
        for z in range(Npiani):
            Drift_max_storey_MS[z]   =  abs(Drift[:step_final_MS,z]).max()
            Drift_max_storey_AS[z]   =  abs(Drift[step_iniz_AS:,z]).max()
            Drift_max_storey_MSAS[z] =  abs(Drift[:,z]).max()
            
            
        Drift_max_MSi   = Drift_max_storey_MS.max()
        Drift_max_ASi   = Drift_max_storey_AS.max()
        Drift_max_MSASi = Drift_max_storey_MSAS.max()    
        
        print("Drift_max_MS = ", Drift_max_MSi)
        print("Drift_max_AS = ", Drift_max_ASi)
        print("Drift_max_MSAS = ", Drift_max_MSASi)
        
        D_max_MSi   = abs(Displ[:step_final_MS,-1]).max()
        D_max_ASi   = abs((Displ[step_iniz_AS:,-1]- Displ[step_final_MS,-1])).max() 
        D_max_MSASi = abs(Displ[:,-1]).max()
        
        RD_MSi   = Displ[step_final_MS,-1]
        RD_ASi   = Displ[-1,-1]-Displ[step_final_MS,-1]
        RD_MSASi = Displ[-1,-1]
        
        
        #%% RISULTATI ###############################################################
        
        E_HYST_MS[MSAS]     = E_HYST_MSi
        Drift_max_MS[MSAS]  = Drift_max_MSi*100
        D_max_MS[MSAS] = D_max_MSi
        RD_MS[MSAS]    = RD_MSi
        
        E_HYST_AS[MSAS]     = E_HYST_ASi
        Drift_max_AS[MSAS]  = Drift_max_ASi*100
        D_max_AS[MSAS] = D_max_ASi
        RD_AS[MSAS]    = RD_ASi
        
        E_HYST_MSAS[MSAS]     = E_HYST_MSASi
        Drift_max_MSAS[MSAS]  = Drift_max_MSASi*100
        D_max_MSAS[MSAS] = D_max_MSASi
        RD_MSAS[MSAS]    = RD_MSASi
        
        
    # print step
    print(MSAS)
    
    
    
#%% PLOT #####################################################################

    pl.figure()
    pl.plot(T,Kin, linewidth=1, color="black",label=r'$E_{K}$')
    pl.plot(T,Kin_Damp, linewidth=1, color="blue",label=r'$E_{K}$'+'+'+r'$E_{D}$')
    pl.plot(T,Kin_Damp_Strain, linewidth=1, color="green",label=r'$E_{K}$'+'+'+r'$E_{D}$'+'+'+r'$E_{E}$')
    pl.plot(T,Ex_work, linewidth=1, color="red",label='Ext. Work')
    
    pl.plot(T[step_final_MS],Ex_work[step_final_MS], 's', color="red")    
    pl.plot(T[step_iniz_AS],Ex_work[step_iniz_AS], 's', color="red")    
    
    
    pl.xlabel('Time [s]', size = 17)
    pl.ylabel('Energy [J]', size = 17)
    pl.legend(bbox_to_anchor=(1.05, 1), edgecolor = 'white', fontsize = 15, markerscale = 1.5, framealpha = 0)
    pl.xticks(size = 15)
    pl.yticks(size = 15)
    
    pl.figure()
    pl.plot(Displ[:,0], Displ[:,1], '-k', lw=2)
    pl.grid()


    



