from NEW_autotest.autotest_rev_parent import Autotest
import pandas as pd
from unifloc.tools import units_converter as uc
from unifloc.common.ambient_temperature_distribution import AmbientTemperatureDistribution
import autotest_core

well_trajectory_data = {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
                                                     data=[[float(0), float(0)],
                                                           [float(1800), float(1800)]])}

table_model_data = {
    'ro': None,
    'rg': None,
    'muo': None,
    'mug': None,
    'bo': None,
    'pb': None,
    'rs': None,
    'z': None,
    'co': None,
    'muw': None,
    'rw': None,
    'stog': None
}
# -------------------------------------------------------------------------------------------
# Для Pipe
# pars_limits = {'wct': [0.0, 0.99],
#                'q_liq': [0.00005787, 0.005787],
#                'rp': [10, 1000],
#                'd': [0.06, 0.3],
#                't_res': [280.15, 380.15],
#                'gamma_oil': [0.65, 0.95]}
#
# limited_pars = {'d': 'tubing',
#                 'roughness': 'tubing'}
# -------------------------------------------------------------------------------------------
# Для газлифта (well)
pars_limits = {'wct': [0.0, 0.99],
                'q_liq': [uc.convert_rate(10, 'm3/day', 'm3/s'),
                          uc.convert_rate(500, 'm3/day', 'm3/s')],
                'rp': [10.0, 1000.0],
                'p_gas_inj': [5000000, 30000000],
                'freq_q_ag': [10000 / 86400, 100000 / 86400],
                'pfl': [10, 50],
                'h_mes': [500, 1100],
                'p_valve': [30 * 101325, 80 * 101325]}

limited_pars = {'h_mes': 'valve3', 'p_valve': 'valve3'}
# -------------------------------------------------------------------------------------------
fluid_data = {"q_fluid": 100 / 86400, "wct": 0,
              "pvt_model_data": {"black_oil": {"gamma_gas": 0.6, "gamma_wat": 1, "gamma_oil": 0.8,
                                               "rp": 50,
                                               "oil_correlations": {"pb": "Standing", "rs": "Standing",
                                                                    "rho": "Standing",
                                                                    "b": "Standing", "mu": "Beggs",
                                                                    "compr": "Vasquez"},
                                               "gas_correlations":
                                                   {"ppc": "Standing", "tpc": "Standing", "z": "Dranchuk",
                                                    "mu": "Lee"},
                                               "water_correlations": {"b": "McCain", "compr": "Kriel",
                                                                      "rho": "Standing",
                                                                      "mu": "McCain"},
                                               # "rsb": {"value": 300, "p": 10000000, "t": 303.15},
                                               # "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
                                               # "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
                                               "table_model_data": None, "use_table_model": False}}}
ambient_temperature_data = {'MD': [0, 1800], 'T': [284.75, 294.75]}
amb_temp = AmbientTemperatureDistribution(ambient_temperature_data)

calc_type = 'well'
# -------------------------------------------------------------------------------------------
# для pipe
# equipment_data = {'packer': True}
# -------------------------------------------------------------------------------------------
# для газлифта
equipment_data = {"gl_system": {
        "valve1": {"h_mes": 1300, "d": 0.003175, "s_bellow": 0.000199677, "s_port": 0.00000838708,
                   "p_valve": uc.convert_pressure(50, "atm", "Pa"),
                   "valve_type": "ЦКсОК"},
        "valve2": {"h_mes": 1100, "d": 0.00396875, "s_bellow": 0.000195483, "s_port": 0.0000129032,
                   "p_valve": uc.convert_pressure(60, "atm", "Pa"),
                   "valve_type": "ЦКсОК"},
        "valve3": {"h_mes": 800, "d": 0.0047625, "s_bellow": 0.000199032, "s_port": 0.0000187096,
                   "p_valve": uc.convert_pressure(40, "atm", "Pa"),
                   "valve_type": "ЦКсОК"}}}
# -------------------------------------------------------------------------------------------
# если рассчитываем газлифт, то тип коррелиции не указывается в pipe_data!!!!
# -------------------------------------------------------------------------------------------
# для pipe
# pipe_data = {'casing': {'bottom_depth': 1800, 'd': 0.33, 'roughness': 0.0001,
#                         'hydr_corr_type': 'beggsbrill',
#                         's_wall': 0.005},
#              'tubing': {'bottom_depth': 1800, 'd': 0.20256, 'roughness': 0.0001,
#                         'hydr_corr_type': 'beggsbrill',
#                         'ambient_temperature_distribution': amb_temp, 's_wall': 0.005
#                         },
#              }
# -------------------------------------------------------------------------------------------
# для газлифта (без указания hydr_corr_type)
pipe_data = {"casing": {"bottom_depth": 1800, "d": 0.146, "roughness": 0.0001, "s_wall": 0.005},
             "tubing": {"bottom_depth": 1400, "d": 0.062, "roughness": 0.0001, "s_wall": 0.005}}
# -------------------------------------------------------------------------------------------
file_path = "C:/Users/PC/PycharmProjects/pythonProject/hagerdon_results/test new autotest.xlsx"
model_path = '504.pips'
pfl = 15
calc_options = {'error_calc': True,
                'save_results': None,
                'plot_results': True}

# если считаем газлифт, то можно задать давление и расход газлифтного газа при закачке
qinj = None  # 100000 / 86400
pinj = None  # 150 * 101325

autotest_core.calc_autotest(p_atma=pfl,
                            file_path=file_path,
                            model_path=model_path,
                            trajectory_data=well_trajectory_data,
                            ambient_temperature_data=ambient_temperature_data,
                            fluid_data=fluid_data,
                            pipe_data=pipe_data,
                            equipment_data=equipment_data,
                            calc_type=calc_type,
                            sample=True,
                            number_of_samples=1,
                            pars_limits=pars_limits,
                            limited_pars=limited_pars,
                            calc_options=calc_options,
                            q_gas_inj=qinj,
                            p_gas_inj=pinj)
