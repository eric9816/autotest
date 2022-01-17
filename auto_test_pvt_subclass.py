"""
Модуль для автотестирования модулей пакета UniflocPy на сходимость с Pipesim.

Позволяет как вызывать сравнительный расчет вручную, так и на рандомных данных в определенных диапазонах данных,
генерируемых с помощью латинского гиперкуба.

20/10/2020

@alexey_vodopyan
"""
# Импорт необходимых библиотека
import pandas as pd
from smt.sampling_methods import LHS
from sixgill.pipesim import Model
from sixgill.definitions import ModelComponents, Parameters, Constants, Units, ProfileVariables
from unifloc.common.ambient_temperature_distribution import AmbientTemperatureDistribution
import copy
import os
import numpy as np
from unifloc.tools import units_converter as uc
from unifloc.pvt.fluid_flow import FluidFlow
from scipy.interpolate import interp1d
from copy import deepcopy
from NEW_autotest.autotest_rev_parent import Autotest


class AutotestPVT(Autotest):

    def __init__(self, model_path,
                 file_path,
                 trajectory_data,
                 ambient_temperature_data,
                 fluid_data,
                 pipe_data,
                 equipment_data):

        super().__init__(model_path,
                         file_path,
                         trajectory_data,
                         ambient_temperature_data,
                         fluid_data,
                         pipe_data,
                         equipment_data)

    def sample_model(self,
                     pars_limits: dict,
                     number_of_samples: int,
                     pfl: float,
                     model_path: str,
                     fluid_data: dict,
                     pipe_data: dict,
                     calc_options: dict,
                     limited_pars: dict = None,
                     equipment_data: dict = None,
                     well_trajectory_data: dict = None,
                     temperature_option: str = None,
                     calc_type: str = 'well',
                     result_path: str = 'results.xlsx',
                     heat_balance: bool = False,
                     flow_direction=None,
                     h_start=None,
                     mean_integral_error=None,
                     ambient_temperature_data=None):
        """
        Функция для расчета моделей на произвольном наборе параметров

        Parameters
        ----------
        :param pars_limits: словарь с параметром и его границами, в которых будет генерироваться данные, dict
        :param limited_pars: словарь с однозначными соответствиями параметров для объектов, которые могут повторяться, dict
            Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing
        :param esp_id: ID-шник насоса из базы насосов, integer
        :param esp_data: база насосов, pd.DataFrame
        :param number_of_samples: количество наборов параметров, integer
        :param pfl: линейное давление, атма, float
        :param freq_q_ag: частота вращения ЭЦН или расход закачки газлифтного газа, Гц или ст. м3/с, float
        :param model_path: путь к файлу с моделью, string
        :param fluid_data: словарь с параметрами флюида, dict
        :param pipe_data: словарь с параметрами труб, dict
        :param equipment_data: словарь с параметрами оборудования, dict
        :param well_trajectory_data: словарь с таблицей с инклинометрией, dict
        :param temperature_option: опция для расчета температуры, 'Const' или 'Linear', string
        :param profile_variables: профильные переменные для вывода в результатах Pipesim, list
        :param calc_type: цель расчета (Что мы хотим посчитать и сравнить? 'well', 'pvt', 'esp', 'pipe'), str
        :param result_path: путь к файлу с результатами, str
        :param heat_balance: True - учет теплопотерь, False - расчет без учета теплопотерь
        :param temperature_option: 'CONST' - темпертарура константа по всей скважине,
                                   'LINEAR' - линейное распределение темпратуры
        :param h_start: 'top' или 'bottom'
        :param flow_direction: -1 - расчет от h_start,
                                1 - расчет к h_start
        :param mean_integral_error: True - среднеинтегральная ошибка,
                                    False - относительная ошибка

        Returns
        -------

        """

        # Подготовка границ для генерации латинского гиперкуба
        keys = list(pars_limits.keys())
        xlimits = [pars_limits[par] for par in keys]

        # Создание латинского гиперкуба
        if isinstance(xlimits, list):
            xlimits = np.array(xlimits)
        sampling = LHS(xlimits=xlimits)

        # Генерация выбранного набора данных
        data = sampling(number_of_samples)

        # Результирующая таблица
        results_df = pd.DataFrame(columns=keys + ['Error', 'Density_inversion_flag'],
                                  index=range(number_of_samples))

        # Итерируемся по набору данных и считаем модели
        for i in range(number_of_samples):

            print(f'Расчет {i + 1} из {number_of_samples}...')

            # Определим, что за параметры и изменим их значения
            pfl_new, freq_q_ag_new, model_path_new, fluid_data_new, pipe_data_new, equipment_data_new = \
                self.find_change_parameter(keys,
                                           data[i, :],
                                           pfl,
                                           model_path,
                                           fluid_data,
                                           pipe_data,
                                           equipment_data,
                                           limited_pars)

            # Передадим в pipesim и проведем расчет
            try:
                results_dict = self.main(well_trajectory_data=well_trajectory_data,
                                         fluid_data=fluid_data_new,
                                         pipe_data=pipe_data_new,
                                         calc_type=calc_type,
                                         equipment_data=equipment_data_new,
                                         pfl=pfl_new,
                                         model_path=model_path_new,
                                         calc_options=calc_options,
                                         temperature_option=temperature_option,
                                         heat_balance=heat_balance,
                                         ambient_temperature_data=ambient_temperature_data,
                                         flow_direction=flow_direction,
                                         h_start=h_start,
                                         mean_integral_error=mean_integral_error)
            except IndexError:
                continue

            # Сохраним значения аргументов
            results_df.loc[i, keys] = data[i, :]

            results_df.loc[i, 'Density_inversion_flag'] = results_dict['density_inversion']

            # Сохраним результаты
            results_df.loc[i, 'Error'] = [results_dict['error_results']]

        # Сконвертируем дебит в м3/сут
        if 'q_fluid' in results_df.columns:
            results_df['q_fluid'] = uc.convert_rate(results_df['q_fluid'], 'm3/s', 'm3/day')

        # Приведем результаты к удобному для вывода формату
        results_df.dropna(how='all', inplace=True)

        if len(results_df) > 0:

            results_df = self.reformat_results(results_df, calc_type)

            results_df.to_excel(result_path)

        return results_df

    def calc_model_uniflocpy(self,
                             fluid_data: dict,
                             equipment_data: dict = None,
                             profile_variables: list = None,
                             calculation_type: str = 'pvt'):
        """
        Унифицированная функция для расчета модели UniflocPy

        Parameters
        ----------
        :param pfl: линейное давление, атма, float
        :param freq_q_ag: частота вращения ЭЦН или расход закачки газлифтного газа, Гц или ст. м3/с, float
        :param fluid_data: словарь с параметрами флюида, dict
        :param pipe_data: словарь с параметрами труб, dict
        :param equipment_data: словарь с параметрами оборудования, dict
        :param well_trajectory_data: словарь с таблицей с инклинометрией, dict
        :param temperature_option: опция для расчета температуры, 'Const' или 'Linear', string
        :param profile_variables: профильные переменные для вывода в результатах Pipesim, list
        :param calculation_type: цель расчета (Что мы хотим посчитать и сравнить? 'well', 'pvt', 'esp', 'pipe'), str

        Returns
        -------

        """
        if equipment_data is None:
            equipment_data = {}

        if profile_variables is None:
            profile_variables = self.profile_variables

        if calculation_type.lower() == 'pvt':
            # Инициализация PVT
            pvt = FluidFlow(**fluid_data)

            # Подготовка листов для параметров
            bo = []
            pb = []
            rs = []
            muo = []
            rho_oil = []
            z = []
            mug = []
            rho_gas = []
            muw = []
            compro = []
            rho_wat = []
            st_oilgas = []
            st_watgas = []
            st_liqgas = []
            q_oil = []
            q_gas = []
            q_wat = []
            q_liq = []
            q_mix = []
            rho_mix = []
            rho_liq = []
            mu_liq = []
            mu_mix = []
            cp_n = []
            cp_l = []

            # Расчет PVT для тех же давлений, что и в Pipesim
            for p, t in zip(self.PRESSURE_INDEX, self.TEMPERATURE_INDEX):
                pvt.calc_flow(p * 10 ** 6, t)
                bo.append(pvt.bo)
                rs.append(pvt.rs)
                muo.append(pvt.muo)
                pb.append(pvt.pb)
                rho_oil.append(pvt.ro)
                z.append(pvt.z)
                mug.append(pvt.mug)
                rho_gas.append(pvt.rg)
                muw.append(pvt.muw)
                compro.append(pvt.co)
                rho_wat.append(pvt.rw)
                st_oilgas.append(pvt.stog)
                st_watgas.append(pvt.stwg)
                st_liqgas.append(pvt.stlg)
                q_oil.append(pvt.qo)
                q_gas.append(pvt.qg)
                q_wat.append(pvt.qw)
                q_liq.append(pvt.ql)
                q_mix.append(pvt.qm)
                rho_mix.append(pvt.rm)
                rho_liq.append(pvt.rl)
                mu_liq.append(pvt.mul)
                mu_mix.append(pvt.mum)
                cp_n.append(pvt.heat_capacity_mixture)
                cp_l.append(pvt.heat_capacity_liq)

            return {'bo': bo,
                    'pb': (np.array(pb) / 10 ** 6).tolist(),
                    'rs': rs,
                    'muo': muo,
                    'ro': rho_oil,
                    'z': z,
                    'mug': mug,
                    'rg': rho_gas,
                    'muw': muw,
                    'co': (np.array(compro) * 10 ** 6).tolist(),
                    'rw': rho_wat,
                    'stog': st_oilgas,
                    'qo': (np.array(q_oil) * 86400).tolist(),
                    'qg': (np.array(q_gas) * 86400).tolist(),
                    'qw': (np.array(q_wat) * 86400).tolist(),
                    # 'q_liq': (np.array(q_liq) * 86400).tolist(),
                    # 'q_mix': (np.array(q_mix) * 86400).tolist(),
                    # 'rho_mix': rho_mix,
                    # 'rho_liq': rho_liq,
                    # 'st_watgas': st_watgas,
                    # 'st_liqgas': st_liqgas,
                    # 'mu_liq': mu_liq,
                    # 'mu_mix': mu_mix,
                    # 'cp_n': cp_n,
                    # 'cp_l': cp_l
                    }

    def calc_model_pipesim(self,
                           pfl: float,
                           model_path: str,
                           fluid_data: dict,
                           pipe_data: dict,
                           equipment_data: dict = None,
                           well_trajectory_data: dict = None,
                           heat_balance=None,
                           temperature_option: str = 'CONST',
                           profile_variables: list = None,
                           flow_direction=None,
                           h_start=None,
                           ambient_temperature_data=None):
        """
        Унифицированная функция для расчета модели Pipesim

        Parameters
        ----------
        :param pfl: линейное давление, П, float
        :param freq_q_ag: частота вращения ЭЦН или расход закачки газлифтного газа, Гц или ст. м3/с, float
        :param model_path: путь к файлу с моделью, string
        :param fluid_data: словарь с параметрами флюида, dict
        :param pipe_data: словарь с параметрами труб, dict
        :param equipment_data: словарь с параметрами оборудования, dict
        :param well_trajectory_data: словарь с таблицей с инклинометрией, dict
        :param temperature_option: опция для расчета температуры, 'Const' или 'Linear', string
        :param profile_variables: профильные переменные для вывода в результатах Pipesim, list

        Returns
        -------

        """
        fluid_data_new = deepcopy(fluid_data)

        if equipment_data is None:
            equipment_data = {}

        if profile_variables is None:
            profile_variables = self.profile_variables

        # Параметры для создания модели
        well_name = 'Test well'
        qliq = fluid_data_new['q_fluid'] * 86400
        wct = fluid_data_new['wct'] * 100

        pvt_model_data = fluid_data_new['pvt_model_data']
        black_oil_model = pvt_model_data['black_oil']
        gamma_oil = black_oil_model['gamma_oil']
        dod = gamma_oil * 1000
        gamma_water = black_oil_model['gamma_wat']
        gamma_gas = black_oil_model['gamma_gas']
        t_res = ambient_temperature_data["T"][1] - 273.15
        gor = black_oil_model['rp']

        h_cas = pipe_data['casing']['bottom_depth']
        d_cas = pipe_data['casing']['d'] * 1000
        roughness_cas = pipe_data['casing']['roughness'] * 1000
        s_wall_cas = 5

        h_tub = pipe_data['tubing']['bottom_depth']
        d_tub = pipe_data['tubing']['d'] * 1000
        roughness_tub = pipe_data['tubing']['roughness'] * 1000
        s_wall_tub = 5
        hydr_corr_type = pipe_data['tubing']['hydr_corr_type']

        # Создадим модель, сохраним и закроем
        model = Model.new(model_path, units=Units.METRIC, overwrite=True)
        model.save()
        model.close()

        # Откроем модель снова, чтобы были метрические единицы измерения
        model = Model.open(model_path, units=Units.METRIC)

        # Добавим скважину в модель
        if flow_direction == 1 and h_start == 'top':
            model.add(ModelComponents.WELL, well_name)
        else:
            print('Что-то не то, включилась нагнеталка')
            model.add(ModelComponents.WELL, well_name, parameters={Parameters.Well.ISINJECTION: True})

        if temperature_option == 'CONST' or temperature_option == 'LINEAR':
            model.set_value(context=well_name, parameter="UseWellHeadAmbientTemperature", value="InputMultipleValues")

        if well_trajectory_data is not None:
            well_trajectory_pipesim = well_trajectory_data['inclinometry'].copy()
            model.set_value(context=well_name, parameter="SurveyType", value="TwoDimensional")
            # Добавим инклинометрию в скважину
            well_trajectory_pipesim.columns = ['MeasuredDepth', 'TrueVerticalDepth']
            model.set_trajectory(context=well_name, value=well_trajectory_pipesim)

        # Установим интервал вывода результата
        model.sim_settings[Parameters.SimulationSetting.PIPESEGMENTATIONMAXREPORTINGINTERVAL] = self.L_REPORT

        # Добавим ЭК в скважину
        model.add(ModelComponents.CASING, "Casing 1", context=well_name,
                  parameters={Parameters.Casing.TOPMEASUREDDEPTH: 0,
                              Parameters.Casing.LENGTH: h_cas,
                              Parameters.Casing.INNERDIAMETER: d_cas,
                              Parameters.Casing.ROUGHNESS: roughness_cas,
                              Parameters.Casing.WALLTHICKNESS: s_wall_cas})

        # Добавим НКТ в скважину
        model.add(ModelComponents.TUBING, "Tubing 1", context=well_name,
                  parameters={Parameters.Tubing.TOPMEASUREDDEPTH: 0,
                              Parameters.Tubing.LENGTH: h_tub,
                              Parameters.Tubing.INNERDIAMETER: d_tub,
                              Parameters.Tubing.ROUGHNESS: roughness_tub,
                              Parameters.Tubing.WALLTHICKNESS: s_wall_tub})

        # Создадим Black-Oil флюид
        model.add(ModelComponents.BLACKOILFLUID, "Black Oil 1",
                  parameters={
                      Parameters.BlackOilFluid.GOR: gor,
                      Parameters.BlackOilFluid.WATERCUT: wct,
                      Parameters.BlackOilFluid.USEDEADOILDENSITY: True,
                      Parameters.BlackOilFluid.DEADOILDENSITY: dod,
                      Parameters.BlackOilFluid.WATERSPECIFICGRAVITY: gamma_water,
                      Parameters.BlackOilFluid.GASSPECIFICGRAVITY: gamma_gas,
                      Parameters.BlackOilFluid.LIVEOILVISCOSITYCORR: "BeggsAndRobinson",
                      Parameters.BlackOilFluid.SinglePointCalibration.SOLUTIONGAS: "Standing",
                      Parameters.BlackOilFluid.SinglePointCalibration.GASCOMPRESSCORRELATION: "Standing",
                      # Parameters.BlackOilFluid.SinglePointCalibration.GASCOMPRESSCORRELATION: "Robinsonetal",
                      Parameters.BlackOilFluid.SinglePointCalibration.OILFVFCORRELATION: "Standing",
                      Parameters.BlackOilFluid.UNDERSATURATEDOILVISCOSITYCORR: "VasquezAndBeggs"
                  })

        parameters = self.pipesim_parameters(wct=wct,
                                             model=model,
                                             black_oil_model=black_oil_model,
                                             heat_balance=heat_balance,
                                             temperature_option=temperature_option,
                                             well_name=well_name,
                                             t_res=t_res,
                                             h_cas=h_cas,
                                             ambient_temperature_data=ambient_temperature_data,
                                             constants=Constants,
                                             equipment_data=equipment_data,
                                             hydr_corr_type=hydr_corr_type,
                                             p=pfl,
                                             qliq=qliq,
                                             h_tub=h_tub,
                                             modelcomponents=ModelComponents)

        if black_oil_model['use_table_model']:
            self.use_table_model(black_oil_model=black_oil_model,
                                 h_cas=h_cas,
                                 parameters=parameters[0],
                                 model=model,
                                 ambient_temperature_data=ambient_temperature_data,
                                 heat_balance_pipesim=parameters[1],
                                 well_name=well_name,
                                 h_tub=h_tub,
                                 pfl=pfl,
                                 t_res=t_res,
                                 fluid_data_new=fluid_data_new)

        results = model.tasks.ptprofilesimulation.run(producer=well_name,
                                                      parameters=parameters[0],
                                                      profile_variables=self.profile_variables)

        model.save()
        model.close()

        global PRESSURE_INDEX
        self.PRESSURE_INDEX = uc.convert_pressure(results.profile[results.cases[0]]['Pressure'], 'bar', 'mpa')[3:]

        global TEMPERATURE_INDEX
        self.TEMPERATURE_INDEX = uc.convert_temperature(results.profile[results.cases[0]]['Temperature'], 'C', 'K')[3:]

        global DEPTH
        self.DEPTH = results.profile[results.cases[0]][ProfileVariables.MEASURED_DEPTH][:1:-1]

        flag = self.analyze_density_inversion(results)

        return results, fluid_data_new, flag

    def main(self,
             well_trajectory_data,
             fluid_data,
             pipe_data,
             calc_type,
             equipment_data,
             pfl,
             model_path,
             calc_options,
             result_path=None,
             heat_balance=None,
             temperature_option=None,
             flow_direction=None,
             h_start=None,
             mean_integral_error=None,
             ambient_temperature_data: object = None
             ) -> dict:
        """
        Основная функция для вызовы всех расчетов

        Parameters
        ----------
        :param well_trajectory_data: словарь с таблицей с инклинометрией, dict
        :param fluid_data: словарь с параметрами флюида, dict
        :param pipe_data: словарь с параметрами труб, dict
        :param calc_type: тип расчета, string, 'well', 'pipe', 'esp', 'pvt'
        :param equipment_data: словарь с параметрами оборудования, dict
        :param freq: частота вращения ЭЦН или расход закачки газлифтного газа, Гц или ст. м3/с
        :param esp_data: DataFrame с данными о насосах, pd.DataFrame
        :param esp_id: ID насоса в базе, integer
        :param pfl: линейное давление, атм, float
        :param stage_by_stage: флаг рассчета по ступеням, boolean
        :param model_path: путь к файлу с моделью, string
        :param result_path: путь к файлу с результатами, string
        :param calc_options: словарь с параметрами расчета, dict0
        :param heat_balance: True - учет теплопотерь, False - расчет без учета теплопотерь
        :param temperature_option: 'CONST' - темпертарура константа по всей скважине,
                                   'LINEAR' - линейное распределение темпратуры
        :param h_start: 'top' или 'bottom'
        :param flow_direction: -1 - расчет от h_start,
                                1 - расчет к h_start
        :param mean_integral_error: True - среднеинтегральная ошибка,
                                    False - относительная ошибка


        Returns
        -------

        """

        if calc_options is None:
            calc_options = {'error_calc': True,
                            'save_results': None,
                            'plot_results': True}

        if equipment_data is not None:
            equipment_data_pipesim = copy.deepcopy(equipment_data)
        else:
            equipment_data_pipesim = equipment_data

        # Запуск расчета на Pipesim
        pipesim_results, fluid_data_new, flag = self.calc_model_pipesim(pfl=uc.convert_pressure(pfl, 'atm', 'pa'),
                                                                        model_path=model_path,
                                                                        fluid_data=fluid_data,
                                                                        pipe_data=pipe_data,
                                                                        heat_balance=heat_balance,
                                                                        equipment_data=equipment_data_pipesim,
                                                                        well_trajectory_data=well_trajectory_data,
                                                                        temperature_option=temperature_option,
                                                                        ambient_temperature_data=ambient_temperature_data,
                                                                        flow_direction=flow_direction,
                                                                        h_start=h_start)

        # Запуск расчета на UniflocPy
        uniflocpy_results = \
            self.calc_model_uniflocpy(fluid_data=fluid_data_new,
                                      equipment_data=equipment_data,
                                      calculation_type=calc_type
                                      )

        # Приведем результаты Pipesim к единому с UniflocPy формату

        pipesim_results_f = self.formate_pipesim_results(pipesim_results, calc_type, list(uniflocpy_results.keys()))

        if calc_options['error_calc']:
            # Запустим расчет ошибки
            error_results = self.calc_error(uniflocpy_results, pipesim_results_f, mean_integral_error)
        else:
            error_results = None

        if calc_options['save_results']:
            # Сохраним результаты
            self.save_results(uniflocpy_results, pipesim_results_f, error_results, result_path, calc_type=calc_type)

        if calc_options['plot_results']:
            # Построим графики
            self.plot_results(uniflocpy_results, pipesim_results_f, calc_type)

        return {'pipesim_results': pipesim_results_f, 'uniflocpy_results': uniflocpy_results,
                'error_results': error_results, 'density_inversion': flag}
#
#
# if __name__ == '__main__':
#     well_trajectory_data = {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
#                                                          data=[[float(0), float(0)],
#                                                                [float(1800), float(1800)]])}
#     table_model_data = {
#         'ro': None,
#         'rg': None,
#         'muo': None,
#         'mug': None,
#         'bo': None,
#         'pb': None,
#         'rs': None,
#         'z': None,
#         'co': None,
#         'muw': None,
#         'rw': None,
#         'stog': None
#     }
#     pars_limits = {'wct': [0.0, 0.99],
#                    'q_liq': [0.00005787, 0.005787],
#                    'rp': [10, 1000],
#                    'd': [0.06, 0.3],
#                    't_res': [280.15, 380.15],
#                    'gamma_oil': [0.65, 0.95]}
#
#     limited_pars = {'d': 'tubing',
#                     'roughness': 'tubing'}
#
#     fluid_data = {'q_fluid': 0.003346, 'wct': 0.84546,
#                   'pvt_model_data': {'black_oil': {'gamma_gas': 0.6, 'gamma_wat': 1, 'gamma_oil': 0.9458,
#                                                    'rp': 918.82,
#                                                    'use_table_model': True, 'table_model_data': table_model_data,
#                                                    }}}
#
#     ambient_temperature_data = {'MD': [0, 1800], 'T': [284.75, 294.75]}
#     amb_temp = AmbientTemperatureDistribution(ambient_temperature_data)
#
#     calc_type = 'pvt'
#     equipment_data = {}
#
#     # pipe_data = {'casing': {'bottom_depth': 1800, 'd': 0.33, 'roughness': 0.0001,
#     #                         'hydr_corr_type': 'hagedornbrown'},
#     #              'tubing': {'bottom_depth': 1800, 'd': 0.20256, 'roughness': 0.0001,
#     #                         'hydr_corr_type': 'hagedornbrown',
#     #                         'ambient_temperature_distribution': amb_temp,
#     #                         },
#     #              }
#     pipe_data = {'casing': {'bottom_depth': 1800, 'd': 0.33, 'roughness': 0.0001,
#                             'hydr_corr_type': 'beggsbrill'},
#                  'tubing': {'bottom_depth': 1800, 'd': 0.20256, 'roughness': 0.0001,
#                             'hydr_corr_type': 'beggsbrill',
#                             'ambient_temperature_distribution': amb_temp,
#                             },
#                  }
#     file_path = "C:/Users/PC/PycharmProjects/pythonProject/hagerdon_results/test new autotest.xlsx"
#     model_path = '504.pips'
#     pfl = 15
#     calc_options = {'error_calc': True,
#                     'save_results': None,
#                     'plot_results': True}
#
#     b = AutotestPVT(model_path, file_path, well_trajectory_data, ambient_temperature_data, fluid_data, pipe_data,
#                     equipment_data)
#
#     b.sample_model(pars_limits=pars_limits,
#                    limited_pars=limited_pars,
#                    number_of_samples=1,
#                    pfl=pfl,
#                    model_path=model_path,
#                    fluid_data=fluid_data,
#                    pipe_data=pipe_data,
#                    equipment_data=equipment_data,
#                    well_trajectory_data=well_trajectory_data,
#                    calc_type=calc_type,
#                    calc_options=calc_options,
#                    result_path=file_path,
#                    heat_balance=False,
#                    temperature_option='CONST',
#                    flow_direction=1,
#                    h_start='top',
#                    mean_integral_error=False,
#                    ambient_temperature_data=ambient_temperature_data)
