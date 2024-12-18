import numpy as np 
from numba import njit, float64, int64
from numba.experimental import jitclass

curr_conv = 0.7 # AUD to USD where necessary
discount_rate = 0.0599 # Real discount rate - same as gencost

## costs come from Apx Table B.9 of GenCost 2023-24 
## year = 2023
#==============================================================================
# utility solar
pv_capex = 1526 # AUD/kW 
pv_fom = 15 # 17 # AUD/kW
pv_vom = 0 # AUD/MWh
pv_lifetime = 30

# onshore wind
wind_ons_capex = 3038 # AUD/kW 
wind_ons_fom = 36 # 25 # AUD/kW 
wind_ons_vom = 0 # AUD/MWh
wind_ons_lifetime = 25

# offhore wind
wind_off_capex = 5545 # AUD/kW 
wind_off_fom = 149.9 # AUD/kW 
wind_off_vom = 0 # AUD/MWh
wind_off_lifetime = 25

## costs unchanged from Lu et al. 2021 https://doi.org/10.1016/j.energy.2020.119678
#==============================================================================
hvdc_overhead_capex = 320 # AUD/MW-km
hvdc_overhead_fom = 3.2 # AUD/MW-km p.a.
hvdc_overhead_vom = 0
hvdc_overhead_lifetime = 50

converter_capex = 160 # AUD/kW
converter_fom = 1.6 # AUD/kW p.a.
converter_vom = 0
converter_lifetime = 30

# undersea costs includer converter
hvdc_undersea_capex = 4000 # AUD/MW-km
hvdc_undersea_fom = 40 # AUD/MW-km p.a.
hvdc_undersea_vom = 0
hvdc_undersea_lifetime = 30

hvac_capex = 1500 # AUD/MW-km
hvac_fom = 15 # AUD/MW-km p.a.
hvac_vom = 0
hvac_lifetime = 50

hydro_purchase = 50 # AUD/MWh p.a.

## costs from re100 cost model - Class A site
#==============================================================================
storage_capexP = 530/0.83 / curr_conv # AUD/kW 
storage_capexE = 47/0.83 / curr_conv # AUD/kWh
storage_fom = 8.21 / curr_conv # AUD/kW p.a.
storage_vom = 0.3 / curr_conv # AUD/MWh p.a.
storage_replace = 112000 / curr_conv # AUD per replace
replace = 50 # every 50 years
storage_lifetime = 100 #operational life

@njit
def annualization_constants(capex, fom, vom, life, dr):
    """ Calculate annualized costs parametrically for power and energy """
    pv = (1-(1+dr)**(-1*life))/dr
    return pow(10,6) * capex / pv + pow(10,6) * fom, vom

@njit
def annualization_transmission_constants(capex, fom, vom, life, d, dr):
    """ Calculate annualized costs parametrically for power and energy, for transmission lines only"""
    pv = (1-(1+dr)**(-1*life))/dr
    return d * capex * pow(10,3) / pv + d * fom * pow(10,3), vom

@njit
def annualization_phes_constants(capex_p, capex_e, fom, vom, replace_cost, replace_life, life, dr):
    """ Calculate annualized costs parametrically for power and energy, for PHES only 
    capex_p, fom: USD/kW
    capex_e: USD/kWh
    vom: USD/MWh
    replace: USD per replace
    replace_life: years """
        
    pv = (1-(1+dr)**(-1*life))/dr
    
    return np.array([
        capex_p* pow(10,6) / pv + fom * pow(10,6), # * GW = cost
        capex_e * pow(10,6) / pv, # * GWh = cost
        vom,# * (MWh discharge p.a.) = cost
        replace_cost * ((1+dr)**(-1*replace_cost) + (1+dr)**(-1*replace_life*2)) / pv, # *1 = cost
        ])


@jitclass([
    ('pv',      float64     ),  
    ('onsw',    float64     ),  
    ('offw',    float64     ),
    ('ac',      float64     ),
    ('hydro',   float64     ),
    ('phes',    float64[:]  ),
    ('hvdc',    float64[:]  ),
    ])
class cost_factors:
    def __init__(self, DClengths, undersea_mask):
        self.pv    = annualization_constants(pv_capex,        pv_fom,        pv_vom,        pv_lifetime,       discount_rate)[0] #vom is 0
        self.onsw  = annualization_constants(wind_ons_capex,  wind_ons_fom,  wind_ons_vom,  wind_ons_lifetime, discount_rate)[0] #vom is 0
        self.offw  = annualization_constants(wind_off_capex,  wind_off_fom,  wind_off_vom,  wind_off_lifetime, discount_rate)[0] #vom is 0
        self.ac    = annualization_transmission_constants(hvac_capex, hvac_fom, hvac_vom, hvac_lifetime, 20, discount_rate)[0] #vom is 0
        
        self.phes  = annualization_phes_constants(storage_capexP, storage_capexE, storage_fom, storage_vom, storage_replace, replace, storage_lifetime, discount_rate)
        
        self.hvdc = np.zeros(len(DClengths), float)
        for i, undersea in enumerate(undersea_mask):
            if undersea:
                self.hvdc[i] = annualization_transmission_constants(hvdc_undersea_capex, hvdc_undersea_fom, hvdc_undersea_vom, hvdc_undersea_lifetime, DClengths[i], discount_rate)[0] # vom is 0
            else: 
                self.hvdc[i] = annualization_transmission_constants(hvdc_overhead_capex, hvdc_overhead_fom, hvdc_overhead_vom, hvdc_overhead_lifetime, DClengths[i], discount_rate)[0]
                self.hvdc[i] += 2*annualization_constants(converter_capex, converter_fom, converter_vom, converter_lifetime, discount_rate)[0]

        self.hydro=hydro_purchase

if __name__ == '__main__':
    from Input import DClengths, undersea_mask
    
    costs = cost_factors(DClengths, undersea_mask)
    
    