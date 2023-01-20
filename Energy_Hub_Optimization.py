# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 23:37:58 2022

@author: schiaffipn
Optimization Model for Energy Hub. 
Six Decision Variables: Number of Turbines, Solar PV Area, Storage Size (Batteries and Hydrogen storage), Installed capacity of Electrolyser & Compressor 
Assumptions and adopted values:
    - Locations
    - Distances to shore
"""
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import os
 

"--------------------Reading Data-------------------"
"---------------------------------------------------"
#Read Input File
cwd = os.getcwd()                                                              #get Current Working Directory        
input_file = os.path.join(cwd, 'input', "input_file.xlsx")                     #create Input File directory

def readExcel (input_file):                                                    #Reading Excel Sheets, use index_col argunment to have the first column as the value of row, then I use loc
 
    df_general = pd.read_excel(input_file,sheet_name="General", header = 0, usecols=[0,1,2],index_col=(0))
    df_economic = pd.read_excel(input_file,sheet_name="Economic", header = 0, usecols=[0,1,2],index_col=(0))
    df_solar = pd.read_excel(input_file, sheet_name="Solar", header = 0, usecols=[0,1,2], index_col=(0))
    df_wind = pd.read_excel(input_file,sheet_name="Wind", header = 0, usecols=[0,1], index_col=(0))
    df_storage = pd.read_excel(input_file, sheet_name="Storage", header = 0, usecols=[0,1,2], index_col=(0))
    df_Hydrogen = pd.read_excel(input_file, sheet_name="Hydrogen", header = 0, usecols=[0,1,2], index_col=(0))
    df_EWind_h = pd.read_excel(input_file,sheet_name="Wind_Power_Data", header = 0, usecols=[0,1,2])
    df_EPV_h = pd.read_excel(input_file,sheet_name="Solar_Power_Data", header = 0, usecols=[0,1])
    df_Export = pd.read_excel(input_file,sheet_name="Export_Capacity", header = 0, usecols=[0,1])
    df_H2_Base_Load = pd.read_excel(input_file,sheet_name="H_BaseLoad", header = 0, usecols=[0,1])
    
    return df_general, df_economic, df_solar, df_wind, df_storage, df_Hydrogen, df_EWind_h, df_EPV_h, df_Export, df_H2_Base_Load

df_general, df_economic, df_solar, df_wind, df_storage, df_Hydrogen, df_EWind_h, df_EPV_h, df_Export, df_H2_Base_Load = readExcel(input_file)

if df_general.loc['hub_config','Input'] == 1:     #Peak shaving  #export as much as the cable capacity 
    print('\n HUB Config = Dedicated Electricity Production')
elif df_general.loc['hub_config','Input'] == 2:   #H2 base load
    print('\n HUB Config = Dedicated Hydrogen Production')                                 
elif df_general.loc['hub_config','Input'] == 3:   
    print('\n HUB Config = Hybrid Energy System')    

"np_calculator(C,r) Function that allows me to get the NPV of a costs or production flow given the rate of return"
"C = the costs or energy production flow, it is a list, or df, indexed over a lifetime (years)"
"r = rate of return "
def np_calculator(C,r): # net present calculator
    net_present = 0
    for k in range(C.size): #range (0 to lifetime)
        net_present = net_present + C[k]/(1+r)**k
    return net_present

"This function creates a flow(array) of costs or energy production using a np array, starting in year 0"
def costs_production_flow(opex_production_asset,lifetime):
    cost_production_flow = np.zeros(lifetime+1) #"OPEX or Production start from year 1 to lifetime --> therefore +1, in year 0 is reserved for CAPEX and Production=0
    cost_production_flow[1:lifetime+1] = opex_production_asset #opex or production starts from year 1.. 
    return cost_production_flow

##I didn't use this table at the end
# def install_monopile(df_general):
#     # Used for calculation of electrolyser and battery structures
#     # (1) Gonzalez-Rodriguez, Angel G. "Review of offshore wind farm cost components." Energy for Sustainable Development 37 (2017): 10-19.
#     # (2) Green, Richard, and Nicholas Vasilakos. "The economics of offshore wind." Energy Policy 39.2 (2011): 496-502.
    
#     # A first representative cost of installation was taken as the average of the values in Table 12 from (1), and divided by 2 to account for the effect of no electrical equipment (as recommended on page 15)
#     cost_installation_per_kW = (115/2)/1000
#     cost_installation = cost_installation_per_kW 
    
#     # The following look-up table adds a multiplier to a platform cost given the water depth and distance to shore, from (2)
#     data = {'0<d<10km'     : [1.00, 1.07, 1.24, 1.40],
#             '10<d<20km'    : [1.02, 1.09, 1.26, 1.43],
#             '20<d<30km'    : [1.04, 1.11, 1.29, 1.46],
#             '30<d<40km'    : [1.07, 1.14, 1.32, 1.49],
#             '40<d<50km'    : [1.09, 1.16, 1.34, 1.52],
#             '50<d<100km'   : [1.18, 1.26, 1.46, 1.65],
#             '100<d<200km'  : [1.41, 1.50, 1.74, 1.97],
#             'd>200km'      : [1.60, 1.71, 1.98, 2.23]}
#     df_multiplier = pd.DataFrame(data, index=["10-20", "20-30", "30-40", "40-50"])
#     df_multiplier.index.name = 'Water Depth [m]'
    
#     if df_general.loc['shore_distance','Input'] > 200:    idx_shore_distance = 7
#     elif df_general.loc['shore_distance','Input'] > 100:  idx_shore_distance = 6
#     elif df_general.loc['shore_distance','Input'] > 50:   idx_shore_distance = 5
#     elif df_general.loc['shore_distance','Input'] > 40:   idx_shore_distance = 4
#     elif df_general.loc['shore_distance','Input'] > 30:   idx_shore_distance = 3
#     elif df_general.loc['shore_distance','Input'] > 20:   idx_shore_distance = 2
#     elif df_general.loc['shore_distance','Input'] > 10:   idx_shore_distance = 1
#     elif df_general.loc['shore_distance','Input'] > 0:    idx_shore_distance = 0
#     if df_general.loc['water_depth','Input'] > 40:        idx_water_depth = 3
#     elif df_general.loc['water_depth','Input'] > 30:      idx_water_depth = 2
#     elif df_general.loc['water_depth','Input'] > 20:      idx_water_depth = 1
#     elif df_general.loc['water_depth','Input'] > 10:      idx_water_depth = 0
    
#     cost_installation = cost_installation * df_multiplier.iloc[idx_water_depth, idx_shore_distance]
    
#     return cost_installation

def power_compressor(df_Hydrogen):
    # Calculate estimated compressor power
    Tmean = 333.15  # Inlet temperature of the compressor
    gamma = 1.4  # Specific heat ratio
    #P_out = Electrolyser['pipeline_inlet']  # Compressor output pressure
    Pout = df_Hydrogen.loc['export_pressure','Input']*100   #bar to kPa
    Pin =  df_Hydrogen.loc['output_pressure','Input']*100   #bar to kPa
    GH = 0.0696
    P_compressor = ((286.76/(GH*0.85*0.98))*Tmean*(gamma/(gamma-1))*(((Pout/Pin)**((gamma-1)/gamma))-1))/3600000     #J to kWh
    return P_compressor

compressor_power = power_compressor(df_Hydrogen)    # for current user input Pin, Pout

#Calculations for the NPC to be used in the Model
#1st Create a cost flow of the OPEX
#2nd If lifetime of asseet is lower than lifetime of the project, then add the corresponing extra costs in the correct year
#3nd Calculate NPC

PV_OPEX_flow = costs_production_flow(df_economic.loc['OPEX_solar_total','Input'],int(df_general.loc['system_lifetime','Input'])) 
np_PV_OPEX = np_calculator(PV_OPEX_flow,df_general.loc['rate_return','Input'])                                  

Wind_OPEX_flow = costs_production_flow(df_economic.loc['OPEX_wind','Input'],int(df_general.loc['system_lifetime','Input'])) 
np_WIND_OPEX= np_calculator(Wind_OPEX_flow,df_general.loc['rate_return','Input']) 

# Storage_CAPEX_installation = install_monopile(df_general) #I did't use the monopile_table at the end
Storage_OPEX_flow = costs_production_flow(df_economic.loc['OPEX_battery','Input'],int(df_general.loc['system_lifetime','Input']))
np_Storage_OPEX = np_calculator(Storage_OPEX_flow,df_general.loc['rate_return','Input'])

# Electrolyser_CAPEX_installation = install_monopile(df_general) #I did't use the monopile_table at the end
Electrolyser_OPEX_flow = costs_production_flow(df_economic.loc['OPEX_Electrolysis','Input'],int(df_general.loc['system_lifetime','Input']))
#Adding Investment costs of the stack over the lifetime
i=1
while df_Hydrogen.loc['lifetime_stack','Input']*i <= df_general.loc['system_lifetime','Input']:
    Electrolyser_OPEX_flow[df_Hydrogen.loc['lifetime_stack','Input']*i]=Electrolyser_OPEX_flow[df_Hydrogen.loc['lifetime_stack','Input']*i] + df_Hydrogen.loc['CAPEX_stack','Input']
    i = i + 1
np_Electrolyser_OPEX = np_calculator(Electrolyser_OPEX_flow,df_general.loc['rate_return','Input'])

Compressor_CAPEX = df_economic.loc['CAPEX_compressor','Input'] * 1.5 #NSE3 Report - Hydrogen transport and compression 
Compressor_OPEX_flow = costs_production_flow(df_economic.loc['OPEX_compressor','Input'],int(df_general.loc['system_lifetime','Input']))
np_Compressor_OPEX = np_calculator(Compressor_OPEX_flow, df_general.loc['rate_return','Input'])

H2_Storage_CAPEX =  df_economic.loc['CAPEX_H2_storage','Input']
H2_Storage_flow =  costs_production_flow(df_economic.loc['OPEX_H2_storage','Input'],int(df_general.loc['system_lifetime','Input']))
np_H2_Storage_OPEX = np_calculator(H2_Storage_flow, df_general.loc['rate_return','Input'])

"---------------------------------PYOMO MODEL---------------------------------"
def CreateModel (df_general, df_economic, df_solar, df_wind, df_storage, df_Hydrogen, df_EWind_h, df_EPV_h, df_Export, df_H2_Base_Load, compressor_power):
    model = pyo.ConcreteModel(name='HPP - Model Optimisation')
    
    #Defining Sets
    # model.T = pyo.Set(ordered = True, initialize=range(8760)) ATTENTION: RangeSet starts in 1!, Range starts in 0! 
    model.T = pyo.Set(initialize = pyo.RangeSet(len(df_EPV_h)), ordered=True) 
    
    "----------------------Parameters----------------------"
    ## Demand as a Cable Capacity [kW] (Export Capacity)
    ##Solar
    if df_solar.loc['config','Input'] == 1: #SOLO
        print ("\n Floating Solar config: SOLO")
        model.PV_CAPEX = pyo.Param(initialize=df_economic.loc['CAPEX_solar_total_SOLO','Input'], mutable=(True))    #In order to perform sensitiviy analisis, the variable has to be saved in a PYOMO Parameter... 
    elif df_solar.loc['config','Input'] == 2: #TOGETHER
        print ("\n Floating Solar config: TOGETHER")
        model.PV_CAPEX = pyo.Param(initialize=df_economic.loc['CAPEX_solar_total_TOGETHER','Input'], mutable=(True))
    elif df_solar.loc['config','Input'] == 3: #SEMI
        print ("\n Floating Solar config: SEMI")
        model.PV_CAPEX = pyo.Param(initialize=df_economic.loc['CAPEX_solar_total_SEMI','Input'], mutable=(True))
    else:
        print ("\n Config Floating Solar - Error")
    
    model.PV_OPEX = pyo.Param(initialize=np_PV_OPEX)
    
    ##Wind Costs
    model.W_CAPEX = pyo.Param(initialize=df_economic.loc['CAPEX_wind','Input'], mutable=(True))
    model.W_OPEX = pyo.Param(initialize=np_WIND_OPEX)
    
    #Storage 
    model.CAPEX_Storage = pyo.Param(initialize=(df_economic.loc['CAPEX_battery','Input']), mutable=(True))
    model.OPEX_Storage = pyo.Param(initialize=np_Storage_OPEX)
    model.charge_rate = pyo.Param(initialize=df_storage.loc['charge_rate','Input'], mutable=(True))
    model.discharge_rate = pyo.Param(initialize=df_storage.loc['discharge_rate','Input'], mutable=(True))   
    
    ##Electrolyser
    model.CAPEX_electrolyser = pyo.Param(initialize=(df_economic.loc['CAPEX_Electrolysis','Input']), mutable=(True))
    model.OPEX_electrolyser = pyo.Param(initialize=np_Electrolyser_OPEX)
    model.electrolyser_efficiency = pyo.Param(initialize=(df_Hydrogen.loc['Electrolysis_Efficiency','Input']), mutable=(True)) #kWh/kg
    
    #Compressor
    model.CAPEX_compressor = pyo.Param(initialize = Compressor_CAPEX)
    model.OPEX_compressor = pyo.Param(initialize = np_Compressor_OPEX)
    model.compressor_power = pyo.Param(initialize = compressor_power)

    #H2 Storage
    model.CAPEX_h2_storage = pyo.Param(initialize=H2_Storage_CAPEX)
    model.OPEX_h2_storage = pyo.Param(initialize = np_H2_Storage_OPEX)


    #Hydrogen Price
    model.H2_price = pyo.Param(initialize = 5, mutable=(True))
    #Electricity Price
    model.electricity_price = pyo.Param(initialize = df_economic.loc['electricity_price','Input'], mutable=(True))    
    
    "--------------------Decision Variables--------------------"
    #Decision Variables
    model.x1 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 667))    # Area PV [ha]
    model.x2 = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0,187)) # Number of Wind Turbines
    model.x3 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,500000))    # Storage Size [kWh]
    model.x4 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,1500000))    # Electrolyser Size [kWh]                       
    model.x5 = pyo.Var(within=pyo.NonNegativeReals)    # Compressor Size  
    model.x6 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,3000000))    # h2 buffer [kg]
    
    "------------------------Variables - the Flows------------------------"
    
    model.EW_h = pyo.Var(model.T,within=pyo.NonNegativeReals)    # Total Wind Energy                 
    model.EPV_h = pyo.Var(model.T,within=pyo.NonNegativeReals)   # Total PV Energy      
    
    model.EUsed_h = pyo.Var(model.T,within=pyo.NonNegativeReals)    #Total Electricity used
    model.ECurtailed_h = pyo.Var(model.T,within=pyo.NonNegativeReals)   #Total electricity Curtailed
    
    model.EUsed_Grid_h = pyo.Var(model.T,within=pyo.NonNegativeReals)        #Electricity Used Wind and PV to Grid  
    model.EUsed_Battery_h = pyo.Var(model.T,within=pyo.NonNegativeReals)     #Electricity from Wind and PV to Storage (Charge_Flow)      
    model.EUsed_H2_h = pyo.Var(model.T,within=pyo.NonNegativeReals)             #Electriity used from W and PV to H2 production 
   
    model.EH2_Electrolyser_h = pyo.Var(model.T,within=pyo.NonNegativeReals)     #Electricity from Wind and PV to Electrolyser 
    model.EH2_Compressor_h = pyo.Var(model.T,within=pyo.NonNegativeReals)       #Electricity from Wind and PV to Compressor 
    
    model.EBattery_Grid_h = pyo.Var(model.T, within=pyo.NonNegativeReals)           #Electricity from baterry to grid (discharge flow)
    model.EBattery_Electrolyser_h = pyo.Var(model.T, within=pyo.NonNegativeReals)   #Electricity from baterry to electrolyser (discharge flow)
    model.EBattery_Compressor_h = pyo.Var(model.T, within=pyo.NonNegativeReals)
    
    
    model.SoC_h = pyo.Var(model.T, within=pyo.NonNegativeReals)                 #Battery SoC
    
    model.EElectrolyser_h = pyo.Var(model.T,within=pyo.NonNegativeReals)        #Total electricity used from PV, Wind and Storage! 
    model.ECompressor_h = pyo.Var(model.T,within=pyo.NonNegativeReals)          #Total electricity used from PV, Wind and Storage!
    model.E_H2_Total_h = pyo.Var(model.T,within=pyo.NonNegativeReals)           
    model.H2_flow_h = pyo.Var(model.T,within=pyo.NonNegativeReals)              #Total H2 production per hour from the Electrolyser
    model.H2_EL_STO_h = pyo.Var(model.T,within=pyo.NonNegativeReals)            #H2 from Electrolyser to storage  
    model.H2_EL_CMP_h = pyo.Var(model.T,within=pyo.NonNegativeReals)            #H2 from electrolyser to CMP--> to pipe
    model.H2_STO_CMP_h = pyo.Var(model.T,within=pyo.NonNegativeReals)           #H2 from Buffer/Storage to CMP--> to pipe
    model.H2_Export_h = pyo.Var(model.T,within=pyo.NonNegativeReals)

    model.h2Bufffer = pyo.Var(model.T,within=pyo.NonNegativeReals)              # H2 Storage

    model.Electricity_export = pyo.Var(model.T,within=pyo.NonNegativeReals)  

# OBJECTIVE FUNCTIONS
    if df_general.loc['hub_config','Input'] == 1:
#OPERATION STRATEGY 1
        def OF (model):                                                                                                                                                                                                                                                                                                                                                                                         #Penalty to energy curtailed
             return ((model.x1*100*100*df_solar.loc['floater_kWm2','Input'])*(model.PV_CAPEX+model.PV_OPEX)) + (model.x2*(model.W_CAPEX+model.W_OPEX)) + (model.x3*(model.CAPEX_Storage + model.OPEX_Storage)) + (model.x4*(model.CAPEX_electrolyser + model.OPEX_electrolyser)) + (model.x5*(model.CAPEX_compressor + model.OPEX_compressor)) + (model.x6*(model.CAPEX_h2_storage + model.OPEX_h2_storage)) + (sum((model.ECurtailed_h[t]*1)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input'])))) #- (sum((model.Electricity_export[t]*model.electricity_price)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input']))))  + (sum((model.E_H2_Total_h[t]*model.electricity_price)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input'])))) - (sum((model.H2_Export_h[t]*model.H2_price)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input'])))) 
        model.ObjFunction = pyo.Objective(rule=OF, sense = pyo.minimize)
    elif df_general.loc['hub_config','Input'] == 2: #H2 Base Load
#OPERATION STRATEGY 2
        def OF (model):                                                                                                                                                                                                                                                                                                                                                                                         #Penalty to energy curtailed
             return ((model.x1*100*100*df_solar.loc['floater_kWm2','Input'])*(model.PV_CAPEX+model.PV_OPEX)) + (model.x2*(model.W_CAPEX+model.W_OPEX)) + (model.x3*(model.CAPEX_Storage + model.OPEX_Storage)) + (model.x4*(model.CAPEX_electrolyser + model.OPEX_electrolyser)) + (model.x5*(model.CAPEX_compressor + model.OPEX_compressor)) + (model.x6*(model.CAPEX_h2_storage + model.OPEX_h2_storage)) + (sum((model.ECurtailed_h[t]*1)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input']))))    
        model.ObjFunction = pyo.Objective(rule=OF, sense = pyo.minimize)        
    elif df_general.loc['hub_config','Input'] == 3: #Free operation
#OPERATION STRATEGY 3
        def OF (model):
             return ((model.x1*100*100*df_solar.loc['floater_kWm2','Input'])*(model.PV_CAPEX+model.PV_OPEX)) + (model.x2*(model.W_CAPEX+model.W_OPEX)) + (model.x3*(model.CAPEX_Storage + model.OPEX_Storage)) + (model.x4*(model.CAPEX_electrolyser + model.OPEX_electrolyser)) + (model.x5*(model.CAPEX_compressor + model.OPEX_compressor)) + (model.x6*(model.CAPEX_h2_storage + model.OPEX_h2_storage)) - (sum((model.H2_Export_h[t]*model.H2_price)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input'])))) - (sum(( model.Electricity_export[t]*model.electricity_price)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input'])))) #+ (sum((model.E_H2_Total_h[t]*model.electricity_price)/((1+df_general.loc['rate_return','Input'])**(i+1)) for t in model.T for i in range(int(df_general.loc['system_lifetime','Input'])))) 
        model.ObjFunction = pyo.Objective(rule=OF, sense = pyo.minimize)

# Electricity Equations      
    #"EPV_h: Total energy generated by PV"    
    def electricity1 (model,t):
        return model.EPV_h[t] == model.x1*df_EPV_h.loc[t-1,'Irradiance 2000-2020']
    #"EW_h: Total energy generated by Wind"
    def electricity2 (model,t):
        return model.EW_h[t] == model.x2*df_EWind_h.loc[t-1,'Wind_Power_1']
    #"Total Used energy is equal to the total generated by PV and Wind minus the energy cutailed"
    def electricity3 (model,t):
        return model.EUsed_h[t] == (model.EPV_h[t] + model.EW_h[t]) - model.ECurtailed_h[t] 
    #"Total Used energy is equal to energy used to the Grid, plus Battery plus Hydrogen production"
    def electricity4 (model,t):
        return model.EUsed_h[t] == model.EUsed_Grid_h[t] + model.EUsed_Battery_h[t] + model.EUsed_H2_h[t]         
      
#SoC Equations 
    #"SoC1: At every hour h, the storage is equal to the storage level at the previous hour plus te charging flows multipliied by the charge rate miinus the discharging flows"
    def SoC1 (model,t):
        # if t == 0:
        if t == 1:
            return  model.SoC_h[t] == 0 # if initial storage is not enough for the sum of the demand in the first hours without irradiance, then the model does not work! in this case the sum is 136.9 kWh #+ (model.charge_flow[t]*df_technical_data.loc['charge_rate','Input']) - (model.discharge_flow[t]/df_technical_data.loc['discharge_rate','Input'])
        else:
            return  model.SoC_h[t] ==  (model.SoC_h[t-1]*df_storage.loc['Self_discharge','Input']) + (model.EUsed_Battery_h[t]*model.charge_rate) - ((model.EBattery_Grid_h[t]+model.EBattery_Electrolyser_h[t]+model.EBattery_Compressor_h[t])/model.discharge_rate)
    #"SoC2: At every hour the storage level has to be lower or equal than the maxium (X3)"
    def SoC2 (model, t):
        return model.SoC_h[t] <= model.x3
    #"SoC3: The discharge flow cannot be higher than the storage level at h-1 (previous hour)"    
    def SoC3 (model, t):
        if t == 1:
            return model.SoC_h[t] == 0
        else:
            return (model.EBattery_Grid_h[t] + model.EBattery_Electrolyser_h[t] + model.EBattery_Compressor_h[t])/model.discharge_rate <= model.SoC_h[t-1]
    #"SoC4: The charge flow cannot be higher than the the maximum capaacity minus SoC prevoius hour"    
    def SoC4 (model, t):
        if t == 1:
            return model.SoC_h[t] == 0
        else:
            return model.EUsed_Battery_h[t]*model.charge_rate <= model.x3 - model.SoC_h[t-1]    
    def SoC5 (model, t):
        return (model.EUsed_Battery_h[t]*model.charge_rate) <= model.x3*df_storage.loc['charge_discharge_power','Input']
    def SoC6 (model, t):
        return ((model.EBattery_Grid_h[t] + model.EBattery_Electrolyser_h[t] + model.EBattery_Compressor_h[t])/model.discharge_rate) <= model.x3*df_storage.loc['charge_discharge_power','Input']
 
    
#Balance Electricity - charasteristic of HUB CONFIG , when congig = 1 then exists Base Electricity load. 
    def balance0 (model,t):    
        return model.Electricity_export[t] == model.EUsed_Grid_h[t] + model.EBattery_Grid_h[t]    
    def balance1 (model,t):    
        return model.Electricity_export[t] >= df_Export.loc[t-1,'Export_Capacity'] 
    def balance2 (model,t):    
        return model.Electricity_export[t] <= df_general.loc['Export_Cable_Capacity','Input'] 

#     if df_general.loc['hub_config','Input'] == 1:
# #OPERATION STRATEGY 1
#         def balance (model,t):                          #Peak shaving export as much as the cable capacity 
#             return (model.EUsed_Grid_h[t] + model.EBattery_Grid_h[t]) >= df_Export.loc[t-1,'Export_Capacity'] #== 0 
#     elif df_general.loc['hub_config','Input'] == 2: #H2 Base Load
# #OPERATION STRATEGY 2 (14-10-22 Added a penalty to the energy curtailed)
#         def balance (model,t):    #H2 Base Load, so total export has to be less than Export Cable capacity
#             return model.EUsed_Grid_h[t] + model.EBattery_Grid_h[t] <= df_Export.loc[t-1,'Export_Capacity']     
#     elif df_general.loc['hub_config','Input'] == 3: #Free operation
# #OPERATION STRATEGY 3
#         def balance (model,t):    #H2 Base Load, so total export has to be less than Export Cable capacity
#             return model.EUsed_Grid_h[t] + model.EBattery_Grid_h[t] <= df_Export.loc[t-1,'Export_Capacity'] 

# Hydrogen Equations
# Total Electricity to Hydrogen production goes to Electrolyser or Compressor
    def hydrogen0 (model, t):
        return  model.EUsed_H2_h[t] == model.EH2_Electrolyser_h[t] + model.EH2_Compressor_h[t]

# Electrolyser Equations
# Total Electricity Consumption Electrolyser
    def electrolyser0 (model, t):
        return  model.EElectrolyser_h[t] == model.EH2_Electrolyser_h[t] + model.EBattery_Electrolyser_h[t]
# Total Hydrogen Production
    def electrolyser1 (model, t):
        return model.H2_flow_h[t] == model.EElectrolyser_h[t]/model.electrolyser_efficiency #kg H2
# Total H2 produced goes to the storage meduim or to the compressor to be exported
    def electrolyser2 (model,t):
        return model.H2_flow_h[t] == model.H2_EL_STO_h[t] + model.H2_EL_CMP_h[t] 
# Electrolyser Capacity
    def electrolyser3 (model, t):
        return model.EElectrolyser_h[t] <= model.x4

#Hydrogen Buffer - Hydrogen Storage
    def h2buffer1 (model, t):
        if t == 1:
            return model.h2Bufffer[t] == 0
        else:
            return model.h2Bufffer[t] == model.h2Bufffer[t-1] + model.H2_EL_STO_h[t] - model.H2_STO_CMP_h[t]
    def h2buffer2 (model, t):
        return model.h2Bufffer[t] <= model.x6
    def h2buffer3 (model, t):
        if t == 1:
            return model.h2Bufffer[t] == 0
        else:
            return model.H2_STO_CMP_h[t] <= model.h2Bufffer[t-1]
    def h2buffer4 (model, t):
        if t == 1:
            return model.h2Bufffer[t] == 0
        else:
            return model.H2_EL_STO_h[t] <= model.x6 - model.h2Bufffer[t-1]        

    def balance1H2 (model,t):
        return model.H2_Export_h[t] == (model.H2_EL_CMP_h[t] + model.H2_STO_CMP_h[t]) 
# Compressor Equations
# Total Electricity Consumption Compressor"
    def compressor0 (model, t):
        return  model.ECompressor_h[t] == model.EH2_Compressor_h[t] + model.EBattery_Compressor_h[t]
# the Total Electricity Consumption Compressor has to be equal to the total H2 being compressed at each time"
    def compressor1 (model, t):
        return (model.H2_Export_h[t])*model.compressor_power == model.ECompressor_h[t]
# Compressor Capacity"
    def compressor2 (model, t):
        return model.ECompressor_h[t] <= model.x5
    
# Balance Hydrogen - charasteristic of HUB CONFIG  
# OPERATION STRATEGY 1 - For this configuration (Peak shaving) thet hydrogen produced is not mandatory - Market Based"
# OPERATION STRATEGY 2 - For this case, H2 Base Load, therefore, the total export has to be higher than H2 Base Load defined" 
# OPERATION STRATEGY 3 - Free Hub -  Electricity and H2 Price based

    def balance2H2 (model,t):
        return model.H2_Export_h[t] >= df_H2_Base_Load.loc[t-1,'H2 Base Load'] #higher than 0 for Peak Shaving or Higer than Base Load 
    def balance3H2 (model,t):
        return model.H2_Export_h[t] <= df_Hydrogen.loc['pipe_capacity','Input']
#Total electricity consumption Hydrogen production
    def H2_electricity_total (model,t):
        return model.E_H2_Total_h[t] == model.ECompressor_h[t]+model.EElectrolyser_h[t] 
        
    model.c1 = pyo.Constraint(model.T, rule=electricity1)
    model.c2 = pyo.Constraint(model.T, rule=electricity2)
    model.c3 = pyo.Constraint(model.T, rule=electricity3)
    model.c4 = pyo.Constraint(model.T, rule=electricity4)
    model.c5 = pyo.Constraint(model.T, rule=SoC1)
    model.c6 = pyo.Constraint(model.T, rule=SoC2)
    model.c7 = pyo.Constraint(model.T, rule=SoC3)
    model.c8 = pyo.Constraint(model.T, rule=SoC4)
    model.c9 = pyo.Constraint(model.T, rule=SoC5)
    model.c10 = pyo.Constraint(model.T, rule=SoC6) 
    model.c11 = pyo.Constraint(model.T, rule=balance0)
    model.c12 = pyo.Constraint(model.T, rule=balance1)
    model.c13 = pyo.Constraint(model.T, rule=balance2) 
    model.c14 = pyo.Constraint(model.T, rule=hydrogen0)
    model.c15 = pyo.Constraint(model.T, rule=electrolyser0)
    model.c16 = pyo.Constraint(model.T, rule=electrolyser1)
    model.c17 = pyo.Constraint(model.T, rule=electrolyser2) 
    model.c18 = pyo.Constraint(model.T, rule=electrolyser3)
    model.c19 = pyo.Constraint(model.T, rule=h2buffer1) 
    model.c20 = pyo.Constraint(model.T, rule=h2buffer2)
    model.c21 = pyo.Constraint(model.T, rule=h2buffer3)
    model.c22 = pyo.Constraint(model.T, rule=h2buffer4)
    model.c23 = pyo.Constraint(model.T, rule=compressor0)
    model.c24 = pyo.Constraint(model.T, rule=compressor1)
    model.c25 = pyo.Constraint(model.T, rule=compressor2)
    model.c26 = pyo.Constraint(model.T, rule=balance1H2)
    model.c27 = pyo.Constraint(model.T, rule=balance2H2)
    model.c28 = pyo.Constraint(model.T, rule=balance3H2)
    model.c29 = pyo.Constraint(model.T, rule=H2_electricity_total)
    
    return model

model = CreateModel(df_general, df_economic, df_solar, df_wind, df_storage, df_Hydrogen, df_EWind_h, df_EPV_h, df_Export, df_H2_Base_Load, compressor_power)      
  
opt = pyo.SolverFactory('gurobi')
results = opt.solve(model, tee=True)
# model.pprint()
results.write()
# model.display()

installed_power_PV = pyo.value(model.x1)*100*100*df_solar.loc['floater_kWm2','Input']

if df_solar.loc['config','Input'] == 1: #SOLO
    results["\n PV Total Area [km2] = "] = installed_power_PV*df_solar.loc['area_SOLO','Input']
elif df_solar.loc['config','Input'] == 2: #TOGETHER
    results["\n PV Total Area [km2] = "] =  installed_power_PV*df_solar.loc['area_TOGETHER','Input']  
elif df_solar.loc['config','Input'] == 3: #SEMI
    results["\n PV Total Area [km2] = "] =  installed_power_PV*df_solar.loc['area_SEMI','Input']  
else:
    print ("\n Config Solar Error")     

print('\n ---------------------------------------------------')
print('\n Decision Variables: ')
print('\n Total net area PV floaters [ha] = ', pyo.value(model.x1))
print('\n Total number of turbines = ', pyo.value(model.x2))
print('\n Total Baterry capacity [kWh] = ', pyo.value(model.x3))
print('\n Electrolyser Capacity [kW] = ', pyo.value(model.x4))
print('\n Compressor Capacity [kW]= ', pyo.value(model.x5))
print('\n H2 Buffer Capacity [kg]= ', pyo.value(model.x6))    


print('\n ---------------------------------------------------')
print('\n Results: ')

print("\n PV Installed Capacity = %4.2f [kWp]" %(installed_power_PV)) 
PV_CAPEX = installed_power_PV*pyo.value(model.PV_CAPEX)
print("\n PV CAPEX = %4.2f [EUR]" %(PV_CAPEX))
PV_OPEX = installed_power_PV*df_economic.loc['OPEX_solar_total','Input']
print("\n PV OPEX = %4.2f [EUR]" %(PV_OPEX))

print("\n Offshore Wind Installed Capacity = %4.2f [kW]" %(pyo.value(model.x2)*df_wind.loc['P_turbine','Input']))
OffshoreWind_CAPEX = pyo.value(model.x2)*pyo.value(model.W_CAPEX)
print("\n Offshore Wind CAPEX = %4.2f [EUR]" %(OffshoreWind_CAPEX))
OffshoreWind_OPEX = pyo.value(model.x2)*df_economic.loc['OPEX_wind','Input']
print("\n Offshore Wind OPEX = %4.2f [EUR]" %(OffshoreWind_OPEX))

Storage_CAPEX = pyo.value(model.x3)*pyo.value(model.CAPEX_Storage)
print("\n Storage CAPEX = %4.2f [EUR]" %(Storage_CAPEX))
Storage_OPEX = pyo.value(model.x3)*df_economic.loc['OPEX_battery','Input']
print("\n Storage OPEX = %4.2f [EUR]" %(Storage_OPEX))

Electrolyser_CAPEX = pyo.value(model.x4)*pyo.value(model.CAPEX_electrolyser)
print("\n Electrolyzer CAPEX = %4.2f [EUR]" %(Electrolyser_CAPEX))
Electrolyser_OPEX = pyo.value(model.x4)*df_economic.loc['OPEX_Electrolysis','Input']
print("\n Electrolyzer OPEX = %4.2f [EUR]" %(Electrolyser_OPEX))

Compressor_CAPEX = pyo.value(model.x5)*pyo.value(model.CAPEX_compressor)
print("\n Compressor CAPEX = %4.2f [EUR]" %(Compressor_CAPEX))
Compressor_OPEX = pyo.value(model.x5)*df_economic.loc['OPEX_compressor','Input']
print("\n Compressor OPEX = %4.2f [EUR]" %(Compressor_OPEX))



#Saving Results in lists to Postprocessing, export to DF, create graphs, etc...:
EPV = []                #Total PV production 
EW = []                 #Total Wind Production 
AEP_h = []              #Total electricity production per hour
E_curtailed = []        #Total Curtailed
E_used = []             #Total Used
EUsed_Grid_h = []
EUsed_Battery_h = []
EUsed_H2_h = []
EUsed_Grid_H2 = []      #for filling graphs  
AE_Export = []
SoC = [] 
E_Export_H2=[]          #for filling graphs               
E_Total_Consumed=[]     #for filling graphs            
E_H2_Total = []         #total electricity used for H2 to be accounted in OPEX!
H2_flow = []            #Total h2 production 
H2_EL_CMP = []
H2_STO_CMP = []
H2_Export = []   
SoC_H2 = []
           
for i in model.T:      
        EPV.append(pyo.value(model.EPV_h[i]))
        EW.append(pyo.value(model.EW_h[i]))  
        AEP_h.append(EW[i-1] + EPV[i-1])                            #Total Electricity Produced
        E_curtailed.append(pyo.value(model.ECurtailed_h[i]))
        E_used.append(pyo.value(model.EUsed_h[i]))                  #Total Electricity Used
        EUsed_Grid_h.append(pyo.value(model.EUsed_Grid_h[i]))       #Used to Grid Directly
        EUsed_Battery_h.append(pyo.value(model.EUsed_Battery_h[i])) #Used to Baterry Directly
        EUsed_H2_h.append(pyo.value(model.EUsed_H2_h[i]))           #Used to H2 Directly
        EUsed_Grid_H2.append(EUsed_Grid_h[i-1] + EUsed_H2_h[i-1])   #Used Grid and H2 (for filling graphs) 
        AE_Export.append(pyo.value(model.EBattery_Grid_h[i])+pyo.value(model.EUsed_Grid_h[i]))        
        E_Export_H2.append(EUsed_H2_h[i-1]+AE_Export[i-1])          #(for filling graphs) electricity to grid direct and from battery plus electricity direct to hydrogen production
        E_H2_Total.append(pyo.value(model.EUsed_H2_h[i])+pyo.value(model.EBattery_Electrolyser_h[i])+pyo.value(model.EBattery_Compressor_h[i]))            
        E_Total_Consumed.append(E_H2_Total[i-1]+AE_Export[i-1])     #(for filling graphs) electricity direct to grid and H2 production + electricity from battery to to grid + CMP + EL         
        H2_flow.append(pyo.value(model.H2_flow_h[i]))
        H2_Export.append(pyo.value(model.H2_Export_h[i]))
        H2_EL_CMP.append(pyo.value(model.H2_EL_CMP_h[i]))
        H2_STO_CMP.append(pyo.value(model.H2_STO_CMP_h[i]))
        SoC.append(pyo.value(model.SoC_h[i])) 
        SoC_H2.append(pyo.value(model.h2Bufffer[i]))       

# Totals coud be find using also: Variable = sum(pyo.value(model.EPV_h[i])+ pyo.value(model.EW_h[i]) for i in model.T)
#Total Wind Production 
print("\n Total wind production [kWh] = ", sum(EW))
#% Curtailed
print('\n Percentage Curtailed = %4.2f' %(sum(E_curtailed)/sum(AEP_h)*100),'%')

#LCOE
AEused =  sum(E_used)
# print('\n AEP [kWh] = ',AEused)
electricity_yearly_production = costs_production_flow(AEused, int(df_general.loc['system_lifetime','Input'])) #yearly production! 
np_AEP = np_calculator(electricity_yearly_production, df_general.loc['rate_return','Input'])

total_OPEX = PV_OPEX + OffshoreWind_OPEX + Storage_OPEX #+ Electrolyser_OPEX + Compressor_OPEX
total_CAPEX = PV_CAPEX + OffshoreWind_CAPEX + Storage_CAPEX #+ Electrolyser_CAPEX + Compressor_CAPEX
total_cost_flow = costs_production_flow(total_OPEX, int(df_general.loc['system_lifetime','Input']))
total_cost_flow[0] = total_CAPEX
np_total_cost = np_calculator(total_cost_flow, df_general.loc['rate_return','Input'])
 
LCOE = np_total_cost/(np_AEP/1000) #EUR/MWh
print('\n LCoE = %4.2f EUR/MWh' %(LCOE))

#LCOH
AEC = sum(E_H2_Total)   #annual electricity consuption
AH2P = sum(H2_flow) #kg
h2_yearly_production = costs_production_flow(AH2P, int(df_general.loc['system_lifetime','Input'])) #yearly production! 
np_production_H2 = np_calculator(h2_yearly_production, df_general.loc['rate_return','Input'])

total_OPEX_AH2P =  Electrolyser_OPEX + Compressor_OPEX  + (AEC*df_economic.loc['electricity_price','Input']) + df_Hydrogen.loc['OPEX_pipe','Input']   #Eprice = 0.077 eur/kWh
total_CAPEX_AH2P = Electrolyser_CAPEX + Compressor_CAPEX + (pyo.value(model.x6)*df_economic.loc['CAPEX_H2_storage','Input']) + df_Hydrogen.loc['CAPEX_pipe','Input'] 

# electricity = (AEC*df_economic.loc['electricity_price','Input'])
# print('\n electriity', electricity)
# print('\n electriity1', total_OPEX_AH2P)
# share_electricity=total_OPEX_AH2P/(AEC*df_economic.loc['electricity_price','Input'])
# print('\n electriity2', share_electricity)

total_cost_flow_H2 = costs_production_flow(total_OPEX_AH2P, int(df_general.loc['system_lifetime','Input']))
total_cost_flow_H2[0] = total_CAPEX_AH2P
#Adding Investment costs of the stack over the lifetime
i=1
while df_Hydrogen.loc['lifetime_stack','Input']*i < df_general.loc['system_lifetime','Input']:
    total_cost_flow_H2[df_Hydrogen.loc['lifetime_stack','Input']*i] = total_cost_flow_H2[df_Hydrogen.loc['lifetime_stack','Input']*i] + (df_Hydrogen.loc['CAPEX_stack','Input']*pyo.value(model.x4))
    i = i + 1    
np_total_cost_H2 = np_calculator(total_cost_flow_H2, df_general.loc['rate_return','Input'])

LCOH = np_total_cost_H2/np_production_H2

print('\n LCoH = %4.2f EUR/kg' %(LCOH))

#VALIDATION! - LCOE
#LCOE
AE_Output = sum(E_used)
electricity_yearly_production = costs_production_flow(AE_Output, int(df_general.loc['system_lifetime','Input'])) #yearly production! 
np_AEUsed = np_calculator(electricity_yearly_production, df_general.loc['rate_return','Input'])

total_OPEX = PV_OPEX + OffshoreWind_OPEX + Storage_OPEX + Electrolyser_OPEX + Compressor_OPEX
total_CAPEX = PV_CAPEX + OffshoreWind_CAPEX + Storage_CAPEX + Electrolyser_CAPEX + Compressor_CAPEX
total_cost_flow = costs_production_flow(total_OPEX, int(df_general.loc['system_lifetime','Input']))
total_cost_flow[0] = total_CAPEX
np_total_cost = np_calculator(total_cost_flow, df_general.loc['rate_return','Input'])
 
LCOE = np_total_cost/(np_AEUsed/1000) #EUR/MWh
print('\n Benchmark LCoE - HPP TNO = %4.2f EUR/MWh' %(LCOE))

#*********PLOT CURVES***********
x = [t for t in model.T]          # time varialbe for the plots 0 to 8759
font_size = 20

#Power Sources
plt.figure(figsize=(12,6))
plt.plot(x[0:8759], EPV[0:8759], label='PV', color='#f1c232') 
plt.plot(x[0:8759], EW[0:8759], label='Wind', color='#2f3c68') 
plt.title("Total Production EWind & EPV", fontsize=font_size)
plt.xlabel("Hours [h]", fontsize=font_size)
plt.ylabel("Power [kWh]", fontsize=font_size)
plt.legend()
plt.grid(True)
plt.show()
#Power Sources
plt.figure(figsize=(12,6))
plt.plot(x[1500:4500], AEP_h[1500:4500], label='PV + Wind') 
plt.title("Total Production EWind + EPV", fontsize=font_size)
plt.xlabel("Hours [h]", fontsize=font_size)
plt.ylabel("Power [kWh]", fontsize=font_size)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(x[2500:3000], EPV[2500:3000], label='PV', color='#f1c232') 
plt.plot(x[2500:3000], EW[2500:3000], label='Wind', color='#2f3c68') 
plt.title("Total Production EWind & EPV", fontsize=font_size)
plt.xlabel("Hours [h]", fontsize=font_size)
plt.ylabel("Power [kWh]", fontsize=font_size)
plt.legend()
plt.grid(True)
plt.show()

#Time frame analysed
tf_start =  0
tf_end = 8760

#Production Electricity Distribution 
fig, (ax1) = plt.subplots(figsize=(12,6))
ax1.plot(x[tf_start:tf_end], AEP_h[tf_start:tf_end],'k',alpha=0.8 ,label='_nolegend_') 
ax1.fill_between(x[tf_start:tf_end],EUsed_Grid_h[tf_start:tf_end], color = '#012d7f', label='Export Cable')
ax1.fill_between(x[tf_start:tf_end],EUsed_Grid_H2[tf_start:tf_end], EUsed_Grid_h[tf_start:tf_end], color = '#38761d', label='Hydrogen Production')
ax1.fill_between(x[tf_start:tf_end],E_used[tf_start:tf_end],EUsed_Grid_H2[tf_start:tf_end], color = 'gold', label='Battery')
ax1.fill_between(x[tf_start:tf_end], AEP_h[tf_start:tf_end],E_used[tf_start:tf_end], color = 'red', label='Curtailed')
ax1.set_title("Produced Electricity Distribution", fontsize=font_size)
ax1.set_xlabel("Hours [h]", fontsize=font_size)
ax1.set_ylabel("Energy [kWh]", fontsize=font_size)
plt.grid(True)
plt.legend()
plt.savefig('filename.png', dpi=300)

#Time frame analysed
tf_start =  2500
tf_end = 3000

#Production Electricity Distribution 
fig, (ax1) = plt.subplots(figsize=(12,6))
ax1.plot(x[tf_start:tf_end], AEP_h[tf_start:tf_end],'k', label='_nolegend_') 
ax1.fill_between(x[tf_start:tf_end],EUsed_Grid_h[tf_start:tf_end], color = '#012d7f', label='Export Cable')
ax1.fill_between(x[tf_start:tf_end],EUsed_Grid_H2[tf_start:tf_end], EUsed_Grid_h[tf_start:tf_end], color = '#38761d', label='Hydrogen Production')
ax1.fill_between(x[tf_start:tf_end],E_used[tf_start:tf_end],EUsed_Grid_H2[tf_start:tf_end], color = 'gold', label='Battery')
ax1.fill_between(x[tf_start:tf_end], AEP_h[tf_start:tf_end],E_used[tf_start:tf_end], color = 'red', label='Curtailed')
ax1.set_title("Produced Electricity Distribution", fontsize=font_size)
ax1.set_xlabel("Hours [h]", fontsize=font_size)
ax1.set_ylabel("Energy [kWh]", fontsize=font_size)
plt.grid(True)
plt.legend()
plt.savefig('filename.png', dpi=300)

#Electricity Distribution Export/Consumption (Grid, Compressor, Hydrogen)
fig, (ax2) = plt.subplots(figsize=(12,6))
ax2.plot(x[tf_start:tf_end], E_Total_Consumed[tf_start:tf_end],'k', label='Total Electricity Used') 
ax2.fill_between(x[tf_start:tf_end],EUsed_Grid_h[tf_start:tf_end], color = '#012d7f', alpha=0.2, label='Export Cable')
ax2.fill_between(x[tf_start:tf_end],AE_Export[tf_start:tf_end], EUsed_Grid_h[tf_start:tf_end], color = '#012d7f', label='From Battery to Cable')
ax2.fill_between(x[tf_start:tf_end],E_Export_H2[tf_start:tf_end], EUsed_Grid_h[tf_start:tf_end], color = '#38761d',alpha=0.2, label='Consumed Hydrogen')
ax2.fill_between(x[tf_start:tf_end],E_Total_Consumed[tf_start:tf_end], E_Export_H2[tf_start:tf_end], color = '#38761d', label='From Battery to Hydrogen')
ax2.set_title("Used Electricity Distribution", fontsize=font_size)
ax2.set_xlabel("Hours [h]", fontsize=font_size)
ax2.set_ylabel("Energy [kWh]", fontsize=font_size)
plt.grid(True)
plt.legend()

#Hydrogen Production
fig, (ax3) = plt.subplots(figsize=(12,6))
ax3.plot(x[tf_start:tf_end],  H2_flow[tf_start:tf_end],color = '#38761d', alpha=0.5, label='Total Hydrogen Production') 
ax3.fill_between(x[tf_start:tf_end],H2_EL_CMP[tf_start:tf_end], color = '#38761d', alpha=0.2, label='from Electrolyser')
ax3.fill_between(x[tf_start:tf_end],H2_Export[tf_start:tf_end], H2_EL_CMP[tf_start:tf_end] ,color = '#38761d', label='from Storage')
ax3.set_title("H2 Production and Exported", fontsize=font_size)
ax3.set_xlabel("Hours [h]", fontsize=font_size)
ax3.set_ylabel("Hydrogen [kg]", fontsize=font_size)
plt.grid(True)
plt.legend()

#SoC
fig, (ax4) = plt.subplots(figsize=(12,6))
ax4.plot(x[tf_start:tf_end],  SoC[tf_start:tf_end],'r--', label='State of Charge') 
ax4.legend()
ax4.set_title("State of Charge", fontsize=font_size)
ax4.set_xlabel("Hours [h]", fontsize=font_size)
ax4.set_ylabel("SoC [kWh]", fontsize=font_size)
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.show()

#Time frame analysed
tf_start =  1000
tf_end = 1500

#Production Electricity Distribution 
fig, (ax1) = plt.subplots(figsize=(12,6))
ax1.plot(x[tf_start:tf_end], AEP_h[tf_start:tf_end],'k', label='_nolegend_') 
ax1.fill_between(x[tf_start:tf_end],EUsed_Grid_h[tf_start:tf_end], color = '#012d7f', label='Export Cable')
ax1.fill_between(x[tf_start:tf_end],EUsed_Grid_H2[tf_start:tf_end], EUsed_Grid_h[tf_start:tf_end], color = '#38761d', label='Hydrogen Production')
ax1.fill_between(x[tf_start:tf_end],E_used[tf_start:tf_end],EUsed_Grid_H2[tf_start:tf_end], color = 'gold', label='Battery')
ax1.fill_between(x[tf_start:tf_end], AEP_h[tf_start:tf_end],E_used[tf_start:tf_end], color = 'red', label='Curtailed')
ax1.set_title("Produced Electricity Distribution", fontsize=font_size)
ax1.set_xlabel("Hours [h]", fontsize=font_size)
ax1.set_ylabel("Energy [kWh]", fontsize=font_size)
plt.grid(True)
plt.legend()

#Electricity Distribution Export/Consumption (Grid, Compressor, Hydrogen)
fig, (ax2) = plt.subplots(figsize=(12,6))
ax2.plot(x[tf_start:tf_end], E_Total_Consumed[tf_start:tf_end],'k', label='Total Electricity Used') 
ax2.fill_between(x[tf_start:tf_end],EUsed_Grid_h[tf_start:tf_end], color = '#012d7f', alpha=0.2, label='Export Cable')
ax2.fill_between(x[tf_start:tf_end],AE_Export[tf_start:tf_end], EUsed_Grid_h[tf_start:tf_end], color = '#012d7f', label='From Battery to Cable')
ax2.fill_between(x[tf_start:tf_end],E_Export_H2[tf_start:tf_end], EUsed_Grid_h[tf_start:tf_end], color = '#38761d',alpha=0.2, label='Consumed Hydrogen')
ax2.fill_between(x[tf_start:tf_end],E_Total_Consumed[tf_start:tf_end], E_Export_H2[tf_start:tf_end], color = '#38761d', label='From Battery to Hydrogen')
ax2.set_title("Used Electricity Distribution", fontsize=font_size)
ax2.set_xlabel("Hours [h]", fontsize=font_size)
ax2.set_ylabel("Energy [kWh]", fontsize=font_size)
plt.grid(True)
plt.legend()

#Hydrogen Production
fig, (ax3) = plt.subplots(figsize=(12,6))
ax3.plot(x[tf_start:tf_end],  H2_flow[tf_start:tf_end],color = '#38761d',linestyle='dashed', alpha=0.3, label='Total Hydrogen Produced') 
ax3.fill_between(x[tf_start:tf_end],H2_EL_CMP[tf_start:tf_end], color = '#38761d', alpha=0.2, label='from Electrolyser')
ax3.fill_between(x[tf_start:tf_end],H2_Export[tf_start:tf_end], H2_EL_CMP[tf_start:tf_end] ,color = '#38761d', label='from Storage')
ax3.set_title("H2 Production and Exported", fontsize=font_size)
ax3.set_xlabel("Hours [h]", fontsize=font_size)
ax3.set_ylabel("Hydrogen [kg]", fontsize=font_size)
plt.grid(True)
plt.legend()

#SoC
fig, (ax4) = plt.subplots(figsize=(12,6))
ax4.plot(x[tf_start:tf_end],  SoC[tf_start:tf_end],'r--', label='State of Charge') 
ax4.legend()
ax4.set_title("State of Charge", fontsize=font_size)
ax4.set_xlabel("Hours [h]", fontsize=font_size)
ax4.set_ylabel("SoC [kWh]", fontsize=font_size)
plt.tight_layout()
plt.grid(True)
plt.legend()


#SoC H2
fig, (ax4) = plt.subplots(figsize=(12,6))
ax4.plot(x[0:8759],  SoC_H2[0:8759],'r--', label='Hydrogen Storage') 
ax4.legend()
ax4.set_title("Hydrogen Storage Level", fontsize=font_size)
ax4.set_xlabel("Hours [h]", fontsize=font_size)
ax4.set_ylabel("Hydrogen Storage [kg]", fontsize=font_size)
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.show()

#Power Sources
plt.figure(figsize=(12,6))
plt.plot(x[1000:1500], EPV[1000:1500], label='PV', color='#f1c232') 
plt.plot(x[1000:1500], EW[1000:1500], label='Wind', color='#2f3c68') 
plt.title("Total Production EWind & EPV", fontsize=font_size)
plt.xlabel("Hours [h]", fontsize=font_size)
plt.ylabel("Power [1e6, kWh]", fontsize=font_size)
plt.legend()
plt.grid(True)
plt.show()


# def LDC (load):
#     return sorted(load, reverse = True)
 
# duration_curve = LDC(AEP_h)



   
# # Sensitivity Analysis  
# Sensitivity={}
# h2_price=[4,4.2,4.4,4.6,4.7,4.8,4.9,5,5.2,5.4,5.6,5.8,6,6.2,6.4,6.6,6.8,7,7.2,7.4,7.5]
# for x in h2_price:
#     model.H2_price=x             # Making the Parameter Mutuable=True, now I change the variable to perform a sensitivity analysis. 
#     results = opt.solve(model) 
#     AEP = sum(pyo.value(model.EPV_h[i]) + pyo.value(model.EW_h[i]) for i in model.T) 
#     curtailed = sum(pyo.value(model.ECurtailed_h[i]) for i in model.T)
#     Sensitivity[x,'% Curtailed']= (curtailed/AEP)*100

# print(Sensitivity)    

# #Creating Graph:
# Y=[Sensitivity[x,'% Curtailed'] for x in h2_price]
# X=[x for x in h2_price]
# plt.plot(X,Y)
# plt.ylabel('% Curtailed')
# plt.xlabel('Hydrogen Price [EUR/kg]')   
# plt.grid(True)
# plt.legend()
# plt.xticks([4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
# plt.show()

# Sensitivity={}
# electricity_price=[0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,0.100,0.105,0.110,0.115,0.120,0.125,0.13]
# for x in electricity_price:
#     model.electricity_price=x             # Making the Parameter Mutuable=True, now I change the variable to perform a sensitivity analysis. 
#     results = opt.solve(model) 
#     AEP = sum(pyo.value(model.EPV_h[i]) + pyo.value(model.EW_h[i]) for i in model.T) 
#     curtailed = sum(pyo.value(model.ECurtailed_h[i]) for i in model.T)
#     Sensitivity[x,'% Curtailed']= (curtailed/AEP)*100

# print(Sensitivity)    

# #Creating Graph:
# Y=[Sensitivity[x,'% Curtailed'] for x in electricity_price]
# X=[x for x in electricity_price]
# plt.plot(X,Y)
# plt.ylabel('% Curtailed')
# plt.xlabel('Electricity Price [EUR/kWh]')   
# plt.grid(True)
# plt.legend()
# plt.xticks([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13])
# plt.show()



# # Sensitivity Analysis Hybrid Energy System Hydrogen Price
# Sensitivity={}
# h2_price=[4,4.2,4.4,4.6,4.7,4.8,4.9,5,5.2,5.4,5.6,5.8,6,6.2,6.4,6.6,6.8,7,7.2,7.4,7.5]
# for x in h2_price:
#     model.H2_price=x             # Making the Parameter Mutuable=True, now I change the variable to perform a sensitivity analysis. 
#     results = opt.solve(model) 
#     Sensitivity[x,'Electrolyser Installed Power']=  pyo.value(model.x4)/1000

# print(Sensitivity)    

# #Creating Graph:
# Y=[Sensitivity[x,'Electrolyser Installed Power'] for x in h2_price]
# X=[x for x in h2_price]
# plt.plot(X,Y)
# plt.ylabel('Electrolyser Installed Power [MW]')
# plt.xlabel('Hydrogen Price [EUR/kg]')   
# plt.grid(True)
# plt.legend()
# plt.xticks([4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
# plt.show()

# # Sensitivity Analysis  Hybrid Energy System Electricity Price
# Sensitivity={}
# electricity_price=[0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,0.100,0.105,0.110,0.115,0.120,0.125,0.13]
# for x in electricity_price:
#     model.electricity_price=x             # Making the Parameter Mutuable=True, now I change the variable to perform a sensitivity analysis. 
#     results = opt.solve(model) 
#     Sensitivity[x,'Electrolyser Installed Power']= pyo.value(model.x4)/1000

# print(Sensitivity)    

# #Creating Graph:
# Y=[Sensitivity[x,'Electrolyser Installed Power'] for x in electricity_price]
# X=[x for x in electricity_price]
# plt.plot(X,Y)
# plt.ylabel('Electrolyser Capacity [MW]', fontsize=14)
# plt.xlabel('Electricity Price [EUR/kWh]', fontsize=14)   
# plt.grid(True)
# plt.legend()
# plt.show()



# df_c1 = pd.DataFrame(c1)   
# df_SoC = pd.DataFrame(SoC)     
# df_Battery_charge = pd.DataFrame(Battery_charge)      
# df_Battery_discharge = pd.DataFrame(Battery_discharge)  
# df_PV_production = pd.DataFrame(PV_production) 
# df_EPV_used = pd.DataFrame(EPV_used) 
# df_PVcurtailment = pd.DataFrame(PVcurtailment) 


#Export to excel

# with pd.ExcelWriter(' Python output.xlsx') as writer:
#     df_c1[0].to_excel(writer, sheet_name="Balance Constraint")
#     df_SoC[0].to_excel(writer, sheet_name="SoC")
#     df_PV_production[0].to_excel(writer, sheet_name="PV Production")
#     df_Battery_charge[0].to_excel(writer, sheet_name="Charge")
#     df_Battery_discharge[0].to_excel(writer, sheet_name="Discharge")
#     df_EPV_used[0].to_excel(writer, sheet_name="PV")
#     df_PVcurtailment[0].to_excel(writer, sheet_name="PV Curtailed")