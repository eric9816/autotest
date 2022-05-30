import pandas as pd
from unifloc.common.ambient_temperature_distribution import AmbientTemperatureDistribution
import autotest_core
from unifloc.tools import units_converter as uc

well_trajectory_data = {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
                                                 data=[[float(0), float(0)],
                                                       [float(1800), float(1800)]])}

# -------------------------------------------------------------------------------------------
# Для Pipe
pars_limits = {'wct': [0.0, 0.99],
               'q_fluid': [1/86400, 500/86400],
               'rp': [1, 1000],
               'd': [0.06, 0.1],
               # 't_res': [280.15, 380.15],
               'gamma_oil': [0.6, 0.8]}

limited_pars = {'d': 'tubing',
                'roughness': 'tubing'}
# -------------------------------------------------------------------------------------------

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
fluid_data = {"q_fluid": 9.94994 / 86400, "wct": 0.00001,
              "pvt_model_data": {"black_oil": {"gamma_gas": 0.6, "gamma_wat": 1, "gamma_oil": 0.668,
                                               "rp": 0.01,
                                               # "oil_correlations": {"pb": "Standing", "rs": "Standing",
                                               #                      "rho": "Standing",
                                               #                      "b": "Standing", "mu": "Beggs",
                                               #                      "compr": "Vasquez", "hc": "const"},
                                               # "gas_correlations":
                                               #     {"ppc": "Standing", "tpc": "Standing", "z": "Standing",
                                               #      "mu": "Lee", "hc": "const"},
                                               # "water_correlations": {"b": "McCain", "compr": "Kriel",
                                               #                        "rho": "Standing",
                                               #                        "mu": "McCain", "hc": "const"},
                                               # "rsb": {"value": 300, "p": 10000000, "t": 303.15},
                                               # "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
                                               # "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
                                               "table_model_data": table_model_data, "use_table_model": False}}}

ambient_temperature_data = {'MD': [0, 1800], 'T': [293.15, 313.15]}
amb_temp = AmbientTemperatureDistribution(ambient_temperature_data)

calc_type = 'pipe'
# -------------------------------------------------------------------------------------------
# для pipe
equipment_data = {'packer': True}
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
pipe_data = {"casing": {"bottom_depth": 1800,
                        "d": 0.146,
                        "roughness": 0.0001,
                        "s_wall": 0.005},
             "tubing": {"bottom_depth": 1800,
                        "d": 0.0924,
                        "roughness": 0.0001,
                        'ambient_temperature_distribution': amb_temp,
                        "s_wall": 0.005}}

# -------------------------------------------------------------------------------------------
file_path = "C:/Users/PC/PycharmProjects/pythonProject/tuffp/1aa tuffp test 1 29.05.xlsx"
model_path = 'C:/Users/PC/PycharmProjects/pythonProject/pips/500.pips'

pfl = 10
calc_options = {'error_calc': True,
                'save_results': False,
                'plot_results': False,
                'scenario': False}

autotest_core.calc_autotest(p_atma=pfl,
                            file_path=file_path,
                            model_path=model_path,
                            trajectory_data=well_trajectory_data,
                            ambient_temperature_data=ambient_temperature_data,
                            fluid_data=fluid_data,
                            data=pipe_data,
                            equipment_data=equipment_data,
                            calc_type=calc_type,
                            sample=True,
                            number_of_samples=1,
                            pars_limits=pars_limits,
                            limited_pars=limited_pars,
                            calc_options=calc_options,
                            hydr_corr_type='beggsbrill',
                            temperature_option='LINEAR',
                            flow_direction=1)





