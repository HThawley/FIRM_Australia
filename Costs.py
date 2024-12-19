import numpy as np 
from numba import njit, float64
from numba.experimental import jitclass

USD_to_AUD = 1.43 # AUD to USD where necessary
discount_rate = 0.0599 # Real discount rate - same as gencost
USD_inflation = 1.18 # 2020->2023
AUD_inflation = 1.16 # 2020->2023


## costs come from Apx Table B.9 of GenCost 2023-24 
## year = 2023
#==============================================================================
# utility solar
csiro_pv = (
    1526,   # capex AUD/kW 
    17,     # fom   AUD/kW p.a.
    0,      # vom   AUD/MWh 
    30,     # life  years
    )
irena_pv = (
    989 * USD_to_AUD,  # capex AUD/kW - AUS average
    18.2 * USD_to_AUD, # fom   AUD/kW - IRENA assumption for OECD, 
    0, 
    30,
    )

# onshore wind
csiro_onsw = (
    3038,   # capex AUD/kW 
    25,     # fom   AUD/kW p.a.
    0,      # vom   AUD/MWh 
    25,     # life  years
    )
irena_onsw = (
    1572 * USD_to_AUD, # capex AUD/kW - AUS average
    42.9 * USD_to_AUD, # fom   AUD/kW p.a. - average of major markets' weighted averages
    0, 
    25
    )

# offshore wind
csiro_offw = (
    5545,   # capex AUD/kW 
    149.9,  # fom   AUD/kW
    0,      # vom   AUD/MWh
    25,     # life  years
    )
irena_offw = (
    2800 * USD_to_AUD, # capex AUD/kW - global weighted average
    92.5 * USD_to_AUD, # fom   AUD/kW p.a. - middle of 2023 range from (Wood Mackenzie, 2023d) pp 123
    0, 
    25,
    )


## costs unchanged from Lu et al. 2021 https://doi.org/10.1016/j.energy.2020.119678
#==============================================================================
hvdc_overhead = (
    320 * AUD_inflation,  # capex AUD/MW-km
    3.2 * AUD_inflation,  # fom   AUD/MW-km p.a.
    0,                    # vom   AUD/MWh-km
    50,                   # life  years
    )

converter = (
    160 * AUD_inflation,  # capex AUD/kw
    1.6 * AUD_inflation,  # fom   AUD/kw p.a.
    0,                    # vom
    30,                   # life  years
    )

# undersea costs includer converter
hvdc_undersea = (
    4000 * AUD_inflation, # capex AUD/kw
    40   * AUD_inflation, # fom   AUD/kw p.a.
    0,                    # vom
    30,                   # life  years
    )

hvac = (
    1500 * AUD_inflation, # capex AUD/MW-km
    15   * AUD_inflation, # fom   AUD/MW-km p.a.
    0,                    # vom 
    50,                   # life  years
    )

hydro_purchase = 50 # AUD/MWh p.a.

## costs from re100 cost model - Class A site
#==============================================================================
phes = (
    530    * USD_inflation * USD_to_AUD, # capex AUD/kW
    47     * USD_inflation * USD_to_AUD, # capex AUD/kWh
    8.21   * USD_inflation * USD_to_AUD, # fom AUD/kW p.a.
    0.3    * USD_inflation * USD_to_AUD, # vom AUD/MWh
    112000 * USD_inflation * USD_to_AUD, # AUD per replace
    50,  # replace lifetime
    100, # life years
    )

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
        capex_p * pow(10,6) / pv + fom * pow(10,6), # * GW = cost
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
    def __init__(self, source, DClengths, undersea_mask):
        if source == 'csiro':
            self.pv    = annualization_constants(*csiro_pv,   discount_rate)[0] #vom is 0
            self.onsw  = annualization_constants(*csiro_onsw, discount_rate)[0] #vom is 0
            self.offw  = annualization_constants(*csiro_offw, discount_rate)[0] #vom is 0
        if source == 'irena':
            self.pv    = annualization_constants(*irena_pv,   discount_rate)[0] #vom is 0
            self.onsw  = annualization_constants(*irena_onsw, discount_rate)[0] #vom is 0
            self.offw  = annualization_constants(*irena_offw, discount_rate)[0] #vom is 0
        
        self.ac    = annualization_transmission_constants(*hvac, 20, discount_rate)[0] #vom is 0
        
        self.phes  = annualization_phes_constants(*phes, discount_rate)
        
        self.hvdc = np.zeros(len(DClengths), float)
        for i, undersea in enumerate(undersea_mask):
            if undersea:
                self.hvdc[i] = annualization_transmission_constants(*hvdc_undersea, DClengths[i], discount_rate)[0] # vom is 0
            else: 
                self.hvdc[i] = annualization_transmission_constants(*hvdc_overhead, DClengths[i], discount_rate)[0] # vom is 0
                self.hvdc[i] += 2*annualization_constants(*converter, discount_rate)[0]

        self.hydro=hydro_purchase

if __name__ == '__main__':
    from Input import costs_source, DClengths, undersea_mask
    
    costs = cost_factors(costs_source, DClengths, undersea_mask)
    
    