"""
Подкласс для автотестирования PVT свойств (pvt)

Позволяет как вызывать сравнительный расчет вручную, так и на рандомных данных в определенных диапазонах данных,
генерируемых с помощью латинского гиперкуба.

31/01/2022

@alexey_vodopyan
@erik_ovsepyan
"""

import pandas as pd
from smt.sampling_methods import LHS
from sixgill.pipesim import Model
from sixgill.definitions import Parameters, ModelComponents, Constants, Units, ProfileVariables, SystemVariables
import copy
import numpy as np
from unifloc.tools import units_converter as uc
from unifloc.pvt.fluid_flow import FluidFlow
import os
from unifloc.equipment.choke import Choke
from copy import deepcopy
from numpy import mean
from NEW_autotest.autotest_class import Autotest
import traceback
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class AutotestChoke(Autotest):
    global CHOKE_INDEX
    global FLAG_CRITICAL
    global SONIC_EC
    global VEL_OUT_EC
    global VEL_IN_EC
    global EXTRA_DP
    global DP
    global FLOWRATE
    global CD
    global CSP
    global CSPG
    global CSPL

    def __init__(self,
                 model_path,
                 file_path,
                 trajectory_data,
                 ambient_temperature_data,
                 fluid_data,
                 choke_data,
                 equipment_data,
                 hydr_corr_type):

        super().__init__(model_path,
                         file_path,
                         trajectory_data,
                         ambient_temperature_data,
                         fluid_data,
                         choke_data,
                         equipment_data,
                         hydr_corr_type)

    def plot_change_param(self, massive_change_param, massive_dp_pipesim, massive_dp_uniflocpy):
        """
        Метод построения графиков
        """
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_ylabel('h, м')
        ax.plot(massive_change_param, massive_dp_pipesim, 'c-', label='Pipesim', linewidth=3)
        ax.plot(massive_change_param, massive_dp_uniflocpy, 'b--', label='UniflocPy', linewidth=3)
        ax.legend()
        plt.show()

    def note_results_to_excel(self, excel_path, pipesim_df, unifloc_py_df, error_df):
        """
        Метод выгрузки рассчитаных pvt-свойств в Pipesim и UniflocPy в excel
        Parameters
        """
        with pd.ExcelWriter(self, excel_path) as writer:
            pipesim_df.to_excel(writer, sheet_name='Pipesim')
            unifloc_py_df.to_excel(writer, sheet_name='UniflocPy')
            error_df.to_excel(writer, sheet_name='Error')

    def calc_error(self, pipesim_df, unifloc_py_df):
        """
        Parameters
        ----------
        :param pipesim_df: массив из Pipesim
        :param unifloc_py_df: массив из UniflocPy

        :return: массив отклонений по каждому pvt-свойству
        -------
        """
        massive_error = pd.DataFrame()

        cols_count_ps = pipesim_df.shape[1]
        cols_count_upy = unifloc_py_df.shape[1]

        cols_ps = list(pipesim_df.columns)
        cols_upy = list(unifloc_py_df.columns)

        pipesim_df = pipesim_df

        pipesim_df_indexes = pipesim_df.index
        unifloc_py_df = unifloc_py_df.loc[pipesim_df_indexes]

        unifloc_py_df.reset_index(drop=True, inplace=True)
        pipesim_df.reset_index(drop=True, inplace=True)

        rows_count = unifloc_py_df.shape[0]

        for i in range(cols_count_ps):
            for j in range(cols_count_upy):
                error = []
                if cols_ps[i] == cols_upy[j]:
                    for row_counter in range(rows_count):
                        error.append(abs(unifloc_py_df[cols_ps[i]][row_counter] - pipesim_df[cols_ps[i]][row_counter]) /
                                     pipesim_df[cols_ps[i]][row_counter] * 100)
                    mean_error = mean(error)
                    error.append(mean_error)
                    massive_error[cols_ps[i]] = error
        massive_error.rename(index={rows_count: 'Среднее отклонение'})
        return massive_error

    # @staticmethod
    def choke_find_change_parameter(self,
                                    keys: list,
                                    data: list,
                                    pfl: float,
                                    model_path: str,
                                    fluid_data: dict,
                                    choke_data: dict,
                                    limited_pars: dict,
                                    t_k: float,
                                    ):
        """
            Функция для обновления значений переменных соответственно значениям латинского гиперкуба

            Parameters
            ----------
            :param keys: названия параметров для которых проводим семплирование, list
            :param data: строка с одним набором параметров, list
            :param pfl: линейное давление, атма, float
            :param model_path: путь к файлу с моделью, string
            :param fluid_data: словарь с параметрами флюида, dict
            :param pipe_data: словарь с параметрами труб, dict
            :param equipment_data: словарь с параметрами оборудования, dict
            :param limited_pars: словарь с однозначными соответствиями параметров для объектов, которые могут повторяться, dict
            Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing

            Returns
            -------

            """
        # Разделим model_path
        model_path_new = model_path[:model_path.find('.pips')]

        pfl_new = None
        fluid_data_new = None
        choke_data_new = None
        t_k_new = None
        init_dicts = [fluid_data, choke_data]

        # Цикл поиска ключей в исходных данных
        for i in range(len(keys)):
            if keys[i] == 'pfl':
                pfl_new = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 't_k':
                t_k = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            else:
                for j in range(len(init_dicts)):

                    # Запустим рекурсивную процедуру поиска ключа по словарю, включая вложенные словари
                    dic_new, flag_find = self.find_key_in_dict(keys[i], data[i], init_dicts[j], limited_pars)

                    # Если ключ найден
                    if flag_find:
                        if j == 0:
                            fluid_data_new = dic_new
                        elif j == 1:
                            choke_data_new = dic_new

                        model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))

                        # Заменим элемент, тк следующий ключ может попасться в этом же словаре
                        init_dicts[j] = dic_new

                        # Нет смысла продолжать поиск, тк каждый ключ уникален
                        break

        # Вернем путь в формат Pipesim
        model_path_new += '.pips'

        # Если данные не обновлялись, значит оставим старые

        if pfl_new is None:
            pfl_new = pfl

        if fluid_data_new is None:
            fluid_data_new = fluid_data

        if choke_data_new is None:
            choke_data_new = choke_data

        if t_k_new is None:
            t_k_new = t_k

        return pfl_new, t_k_new, model_path_new, fluid_data_new, choke_data_new

    def choke_reformat_results(self,
                               df,
                               ):
        """
        Функция правильного форматирования результатов многовариантного тестирования

        Parameters
        ----------
        :param df: DataFrame с результатами
        :param calc_type: тип расчета

        Returns
        -------

        """
        for i, row in df.iterrows():
            error_dict = row['Error']

            for k, v in error_dict.items():
                if v == 't2pip__t1__t2un':
                    continue
                else:
                    df.loc[i, 'Error ' + k] = v

        del df['Error']
        return df

    def sample_model(self,
                     pars_limits: dict,
                     number_of_samples: int,
                     p1: float,
                     t_k: float,
                     model_path: str,
                     fluid_data: dict,
                     choke_data: dict,
                     limited_pars: dict = None,
                     calc_type: str = 'choke',
                     file_path: str = 'results.xlsx',
                     flow_direction=None,
                     ):
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
        results_df = pd.DataFrame(columns=keys,
                                  index=range(number_of_samples))

        # Итерируемся по набору данных и считаем модели
        for i in range(number_of_samples):

            print(f'Расчет {i + 1} из {number_of_samples}...')

            # Определим, что за параметры и изменим их значения
            p1_new, t_k_new, model_path_new, fluid_data_new, choke_data_new, = \
                self.choke_find_change_parameter(keys,
                                                 data[i, :],
                                                 p1,
                                                 model_path,
                                                 fluid_data,
                                                 choke_data,
                                                 limited_pars,
                                                 t_k)

            # Передадим в pipesim и проведем расчет
            try:
                results_dict = self.main(p1=p1_new,
                                         t1=t_k_new,
                                         choke_data=choke_data_new,
                                         fluid_data=fluid_data_new,
                                         model_path=model_path_new,
                                         flow_direction=flow_direction,
                                         )
            except IndexError:
                continue

            # Сохраним значения аргументов
            results_df.loc[i, keys] = data[i, :]

            # results_df.loc[i, 'Density_inversion_flag'] = results_dict['density_inversion']

            # Сохраним результаты
            results_df.loc[i, 'Error'] = [results_dict['error_results']]

        # Сконвертируем дебит в м3/сут
        if 'q_fluid' in results_df.columns:
            results_df['q_fluid'] = uc.convert_rate(results_df['q_fluid'], 'm3/s', 'm3/day')

        # Приведем результаты к удобному для вывода формату
        results_df.dropna(how='all', inplace=True)

        if len(results_df) > 0:
            results_df = self.choke_reformat_results(results_df)

            results_df.to_excel(file_path)

        return results_df

    def calc_model_uniflocpy(self,
                             p1,
                             t1,
                             choke_data: dict,
                             fluid_data: dict,
                             flow_direction: int = 1,
                             ):
        step_num = 0
        d_choke = choke_data['d_choke']
        d_up = choke_data['d_up']
        qliq = fluid_data['q_fluid']
        wct = fluid_data['wct']
        rp = fluid_data['pvt_model_data']['black_oil']['rp']
        pvt_model_data = fluid_data['pvt_model_data']
        fluid_flow = FluidFlow(qliq, wct, pvt_model_data)
        if choke_data['temperature_drop']:
            choke = Choke(0, d_choke, d_up, fluid=fluid_flow, temperature_drop=True)
        else:
            choke = Choke(0, d_choke, d_up, fluid=fluid_flow, temperature_drop=False)

        dp_array_uniflocpy = {}

        try:
            pt_results = choke.calc_pt(p1, t1, flow_direction, c_choke=None)
            gf = choke.fluid.gf
            p2 = pt_results[0]
            t2 = pt_results[1]
            q_liq = choke.fluid.ql
            q_gas = choke.fluid.qg

            dp_array_uniflocpy['p2'] = p2
            dp_array_uniflocpy['t2'] = t2
            dp_array_uniflocpy['p1'] = p1
            dp_array_uniflocpy['t1'] = t1
            dp_array_uniflocpy['gf'] = gf
            dp_array_uniflocpy['regime'] = choke.regime_type
            dp_array_uniflocpy['ql'] = q_liq
            dp_array_uniflocpy['qg'] = q_gas
            # dp_array_uniflocpy['dt'] = t1 - t2
            # dp_array_uniflocpy['t_check'] = None
            # dp_array_uniflocpy[step_num, 4] = choke.
            # if choke.regime_type == 'critical' or choke.regime_type == 'supercritical':
            #     dp_array_uniflocpy[step_num, 5] = 1
            # else:
            #     dp_array_uniflocpy[step_num, 5] = 0
            # dp_array_uniflocpy[step_num, 6] = choke.fluid_velocity
            # dp_array_uniflocpy['p_crit'] = choke.
            # dp_array_uniflocpy['extra_dp'] = choke.extra_dp
            # dp_array_uniflocpy['wct'] = wct
            # dp_array_uniflocpy['rp'] = rp
            # dp_array_uniflocpy['qliq'] = qliq
            # dp_array_uniflocpy['d_choke'] = d_choke
            # dp_array_uniflocpy['cd'] = choke_data['cd']
            # dp_array_uniflocpy[step_num, 14] = CD
            # dp_array_uniflocpy['c_vg'] = choke.c_vg
            # dp_array_uniflocpy['c_v'] = choke.c_vg
            # dp_array_uniflocpy['c_vl'] = choke.c_vl

        except:
            dp_array_uniflocpy = {}
            print(traceback.format_exc())

        return dp_array_uniflocpy, t2

    def calc_model_pipesim(self,
                           p1: float,
                           t1: float,
                           model_path: str,
                           fluid_data: dict,
                           choke_data: dict,
                           flow_direction: int = 1,
                           cd=None):

        # Принимаем исходные данные о флюиде
        q_liq = fluid_data['q_fluid'] * 86400

        wct = fluid_data['wct'] * 100

        pvt_model_data = fluid_data['pvt_model_data']
        black_oil_model = pvt_model_data['black_oil']

        gamma_oil = black_oil_model['gamma_oil']
        dod = gamma_oil * 1000
        gamma_water = black_oil_model['gamma_wat']
        gamma_gas = black_oil_model['gamma_gas']
        gor = black_oil_model['rp']
        flag_calibr = False
        d_tub = choke_data['d_up'] * 1000
        delta_wall_tub = choke_data['delta_wall_tub']
        roughness = choke_data['roughness']
        d_choke = choke_data['d_choke'] * 1000
        t1 = t1 - 273.15

        # Калибровочные значения параметров rsb, bob, muob
        if 'rsb' in black_oil_model.keys():
            rsb_dict = black_oil_model['rsb']
            if rsb_dict is not None:
                flag_calibr = True
                rsb_p = uc.convert_pressure(rsb_dict['p'], 'pa', 'bar')
                rsb_t = rsb_dict['t'] - 273.15
                rsb_value = rsb_dict['value']

        if 'muob' in black_oil_model.keys():
            muo_dict = black_oil_model['muob']
            if muo_dict is not None:
                flag_calibr = True
                muo_p = uc.convert_pressure(muo_dict['p'], 'pa', 'bar')
                muo_t = muo_dict['t'] - 273.15
                muo_value = muo_dict['value']

        if 'bob' in black_oil_model.keys():
            bob_dict = black_oil_model['bob']
            if bob_dict is not None:
                flag_calibr = True
                bob_p = uc.convert_pressure(bob_dict['p'], 'pa', 'bar')
                bob_t = bob_dict['t'] - 273.15
                bob_value = bob_dict['value']

        p1 = uc.convert_pressure(p1, 'pa', 'bar')
        # if p_wh is not None:
        #     p_wh = uc.convert_pressure(p_wh, 'pa', 'bar')

        # Задаем шаг расчета
        L_REPORT = 1

        system_variables = [
            SystemVariables.PRESSURE,
            SystemVariables.TEMPERATURE]

        profile_variables = [
            ProfileVariables.TEMPERATURE,
            ProfileVariables.PRESSURE,
            ProfileVariables.MASS_FLOWRATE_OIL_INSITU,
            ProfileVariables.MASS_FLOWRATE_GAS_INSITU,
            ProfileVariables.MASS_FLOWRATE_WATER_INSITU,
            ProfileVariables.VOLUME_FLOWRATE_OIL_INSITU,
            ProfileVariables.VOLUME_FLOWRATE_GAS_INSITU,
            ProfileVariables.VOLUME_FLOWRATE_WATER_INSITU,
            ProfileVariables.VOLUME_FLOWRATE_LIQUID_INSITU,
            ProfileVariables.DENSITY_OIL_INSITU,
            ProfileVariables.DENSITY_GAS_INSITU,
            ProfileVariables.DENSITY_WATER_INSITU,
            ProfileVariables.DENSITY_LIQUID_INSITU,
            ProfileVariables.HEAT_CAPACITY_FLUID_INSITU,
            ProfileVariables.Z_FACTOR_GAS_INSITU,
            ProfileVariables.BUBBLE_POINT_PRESSURE_INSITU,
            ProfileVariables.SONIC_VELOCITY_IN_FLUID,
            ProfileVariables.VOLUME_FRACTION_GAS_INSITU,
            ProfileVariables.TEMPERATURE_GRADIENT_JOULE_THOMSON,
            ProfileVariables.PRESSURE_GRADIENT_TOTAL,
            ProfileVariables.DENSITY_FLUID_NO_SLIP_INSITU,
        ]

        model = Model.new(model_path, units=Units.METRIC, overwrite=True)
        model.save()
        model.close()

        model = Model.open(model_path, units=Units.METRIC)
        model.save()

        # Создадим Black-Oil модель флюида
        if flag_calibr:
            model.add(ModelComponents.BLACKOILFLUID, "Black Oil 1",
                      parameters={
                          Parameters.BlackOilFluid.GOR: gor,
                          Parameters.BlackOilFluid.WATERCUT: wct,
                          # Parameters.BlackOilFluid.USEGASRATIO: 'GLR',
                          Parameters.BlackOilFluid.USEDEADOILDENSITY: True,
                          Parameters.BlackOilFluid.DEADOILDENSITY: dod,
                          Parameters.BlackOilFluid.WATERSPECIFICGRAVITY: gamma_water,
                          Parameters.BlackOilFluid.GASSPECIFICGRAVITY: gamma_gas,
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPOFVF_VALUE: bob_value,
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPOFVF_PRESSURE: bob_p,
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPOFVF_TEMPERATURE: bob_t,
                          Parameters.BlackOilFluid.SinglePointCalibration.BUBBLEPOINTSATGAS_VALUE: rsb_value,
                          Parameters.BlackOilFluid.SinglePointCalibration.BUBBLEPOINTSATGAS_PRESSURE: rsb_p,
                          Parameters.BlackOilFluid.SinglePointCalibration.BUBBLEPOINTSATGAS_TEMPERATURE: rsb_t,
                          Parameters.BlackOilFluid.SinglePointCalibration.ABOVEBBPTYPE: "OFVF",
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPTYPE: "OFVF",
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPLIVEOILVISCOSITY_VALUE: muo_value,
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPLIVEOILVISCOSITY_TEMPERATURE: muo_t,
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPLIVEOILVISCOSITY_PRESSURE: muo_p,
                          Parameters.BlackOilFluid.SinglePointCalibration.LIVEOILVISCCORRELATION: "BeggsAndRobinson",
                          Parameters.BlackOilFluid.LIVEOILVISCOSITYCORR: "BeggsAndRobinson",
                          Parameters.BlackOilFluid.SinglePointCalibration.SOLUTIONGAS: "Standing",
                          Parameters.BlackOilFluid.SinglePointCalibration.GASCOMPRESSCORRELATION: "Standing",
                          Parameters.BlackOilFluid.SinglePointCalibration.OILFVFCORRELATION: "Standing",
                          Parameters.BlackOilFluid.UNDERSATURATEDOILVISCOSITYCORR: "VasquezAndBeggs",
                          # Parameters.BlackOilFluid.ThermalData.OILHEATCAPACITY: 1884.054,
                          # Parameters.BlackOilFluid.ThermalData.GASHEATCAPACITY: 2302.733,
                          # Parameters.BlackOilFluid.ThermalData.WATERHEATCAPACITY: 4186.787,
                          # Parameters.BlackOilFluid.DEADOILVISCOSITYCORR: 'User2Point',
                          # Parameters.BlackOilFluid.DEADOILTEMPERATURE1: 21.11111,
                          # Parameters.BlackOilFluid.DEADOILVISCOSITY1: 1.964284,
                          # Parameters.BlackOilFluid.DEADOILTEMPERATURE2: 26.6667,
                          # Parameters.BlackOilFluid.DEADOILVISCOSITY2: 0.6592401

                      })
        else:
            model.add(ModelComponents.BLACKOILFLUID, "Black Oil 1",
                      parameters={
                          Parameters.BlackOilFluid.GOR: gor,
                          # Parameters.BlackOilFluid.USEGASRATIO: 'GLR',
                          # Parameters.BlackOilFluid.GLR: gor,
                          Parameters.BlackOilFluid.WATERCUT: wct,
                          Parameters.BlackOilFluid.USEDEADOILDENSITY: True,
                          Parameters.BlackOilFluid.DEADOILDENSITY: dod,
                          Parameters.BlackOilFluid.WATERSPECIFICGRAVITY: gamma_water,
                          Parameters.BlackOilFluid.GASSPECIFICGRAVITY: gamma_gas,
                          Parameters.BlackOilFluid.SinglePointCalibration.ABOVEBBPTYPE: "OFVF",
                          Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPTYPE: "OFVF",
                          Parameters.BlackOilFluid.SinglePointCalibration.LIVEOILVISCCORRELATION: "BeggsAndRobinson",
                          Parameters.BlackOilFluid.LIVEOILVISCOSITYCORR: "BeggsAndRobinson",
                          Parameters.BlackOilFluid.SinglePointCalibration.SOLUTIONGAS: "Standing",
                          Parameters.BlackOilFluid.SinglePointCalibration.GASCOMPRESSCORRELATION: "Standing",
                          Parameters.BlackOilFluid.SinglePointCalibration.OILFVFCORRELATION: "Standing",
                          Parameters.BlackOilFluid.UNDERSATURATEDOILVISCOSITYCORR: "VasquezAndBeggs",
                          # Parameters.BlackOilFluid.ThermalData.OILHEATCAPACITY: 1884.054 / 1000,
                          # Parameters.BlackOilFluid.ThermalData.GASHEATCAPACITY: 2302.733 / 1000,
                          # Parameters.BlackOilFluid.ThermalData.WATERHEATCAPACITY: 4186.787 / 1000,
                      })
        # model = Model.open(model_path, units=Units.METRIC)
        model.save()

        model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = 'HEAT BALANCE = OFF'
        model.sim_settings[Parameters.SimulationSetting.AMBIENTTEMPERATURE] = t1

        # Добавим источник в модель
        model.add(ModelComponents.SOURCE, 'Start',
                  parameters={Parameters.Source.ASSOCIATEDBLACKOILFLUID: 'Black Oil 1',
                              Parameters.Source.TEMPERATURE: t1,
                              # Parameters.Source.LIQUIDFLOWRATE: q_liq,
                              # Parameters.Source.PRESSURE: p_fl
                              })

        # Добавим ТП 1 в модель

        model.add(ModelComponents.FLOWLINE, 'FL1',
                  parameters={Parameters.Flowline.DETAILEDMODEL: False,
                              Parameters.Flowline.USEENVIRONMENTALDATA: True,
                              Parameters.Flowline.INNERDIAMETER: d_tub,
                              Parameters.Flowline.WALLTHICKNESS: delta_wall_tub,
                              Parameters.Flowline.ROUGHNESS: roughness,
                              # Parameters.Flowline.LENGTH: 100,
                              Parameters.Flowline.HORIZONTALDISTANCE: 1,
                              Parameters.Flowline.UNDULATIONRATE: 0,
                              Parameters.Flowline.ELEVATIONDIFFERENCE: 0,
                              Parameters.Flowline.AMBIENTTEMPERATURE: t1
                              })

        # if cd is not None:
        #     sub_crit_corr = "Mechanistic"
        # else:
        #     cd = 0
        sub_crit_corr = "Mechanistic"
        # Добавим штуцер в скважину
        model.add(ModelComponents.CHOKE, "Choke 1",
                  parameters={Parameters.Choke.BEANSIZE: d_choke,
                              Parameters.Choke.SUBCRITICALCORRELATION: sub_crit_corr,
                              Parameters.Choke.DISCHARGECOEFFICIENT: cd,
                              Parameters.Choke.CRITICALCORRELATION: "Mechanistic",
                              Parameters.Choke.UPSTREAMPIPEID: d_tub,
                              Parameters.Choke.USEFLOWRATEFORCRITICALFLOW: False,
                              Parameters.Choke.USEPRESSURERATIOFORCRITICALFLOW: False,
                              Parameters.Choke.USESONICDOWNSTREAMVELOCITYFORCRITICALFLOW: True,
                              Parameters.Choke.USESONICUPSTREAMVELOCITYFORCRITICALFLOW: False,
                              Parameters.Choke.HEATCAPACITYRATIO: 1.26,
                              Parameters.Choke.PRINTDETAILEDCALCULATIONS: True,
                              Parameters.Choke.ADJUSTSUBCRITICALCORRELATION: False
                              })

        # Добавим ТП 2 в модель
        model.add(ModelComponents.FLOWLINE, 'FL2',
                  parameters={Parameters.Flowline.DETAILEDMODEL: False,
                              Parameters.Flowline.USEENVIRONMENTALDATA: True,
                              Parameters.Flowline.INNERDIAMETER: d_tub,
                              Parameters.Flowline.WALLTHICKNESS: delta_wall_tub,
                              Parameters.Flowline.ROUGHNESS: roughness,
                              # Parameters.Flowline.LENGTH: 100,
                              Parameters.Flowline.HORIZONTALDISTANCE: 1,
                              Parameters.Flowline.UNDULATIONRATE: 0,
                              Parameters.Flowline.ELEVATIONDIFFERENCE: 0,
                              Parameters.Flowline.AMBIENTTEMPERATURE: t1
                              })
        model.save()
        # Добавим Junction в модель
        model.add(ModelComponents.JUNCTION, 'J',
                  parameters={Parameters.Junction.TREATASSOURCE: True,
                              # Parameters.Junction.ASSOCIATEDBLACKOILFLUID: 'Black Oil 1',
                              # Parameters.Junction.PRESSURE: p_fl,
                              # Parameters.Junction.TEMPERATURE: t_fl,
                              # Parameters.Junction.LIQUIDFLOWRATE: q_liq
                              })

        model.connect({ModelComponents.SOURCE: 'Start'}, {ModelComponents.FLOWLINE: 'FL1'})
        model.connect({ModelComponents.FLOWLINE: 'FL1'}, {ModelComponents.CHOKE: 'Choke 1'})
        model.connect({ModelComponents.CHOKE: 'Choke 1'}, {ModelComponents.FLOWLINE: 'FL2'})
        model.connect({ModelComponents.FLOWLINE: 'FL2'}, {ModelComponents.JUNCTION: 'J'})

        parameters = {Parameters.PTProfileSimulation.OUTLETPRESSURE: p1,
                      Parameters.PTProfileSimulation.LIQUIDFLOWRATE: q_liq,
                      Parameters.PTProfileSimulation.FLOWRATETYPE: Constants.FlowRateType.LIQUIDFLOWRATE,
                      Parameters.PTProfileSimulation.CALCULATEDVARIABLE: Constants.CalculatedVariable.INLETPRESSURE}

        # Установим интервал вывода результата
        model.sim_settings[Parameters.SimulationSetting.PIPESEGMENTATIONMAXREPORTINGINTERVAL] = L_REPORT

        model.save()

        if flow_direction == -1:
            results = model.tasks.ptprofilesimulation.run(producer='Start',
                                                          parameters={
                                                              Parameters.PTProfileSimulation.INLETPRESSURE: p1,
                                                              Parameters.PTProfileSimulation.LIQUIDFLOWRATE: q_liq,
                                                              Parameters.PTProfileSimulation.FLOWRATETYPE: Constants.FlowRateType.LIQUIDFLOWRATE,
                                                              Parameters.PTProfileSimulation.CALCULATEDVARIABLE: Constants.CalculatedVariable.OUTLETPRESSURE},
                                                          system_variables=system_variables,
                                                          profile_variables=self.profile_variables)
        elif flow_direction == 1:
            results = model.tasks.ptprofilesimulation.run(producer='Start',
                                                          parameters={
                                                              Parameters.PTProfileSimulation.OUTLETPRESSURE: p1,
                                                              Parameters.PTProfileSimulation.LIQUIDFLOWRATE: q_liq,
                                                              Parameters.PTProfileSimulation.FLOWRATETYPE: Constants.FlowRateType.LIQUIDFLOWRATE,
                                                              Parameters.PTProfileSimulation.CALCULATEDVARIABLE: Constants.CalculatedVariable.INLETPRESSURE},
                                                          system_variables=system_variables,
                                                          profile_variables=self.profile_variables)
        # results = model.tasks.ptprofilesimulation.run(producer='Start',
        #                                               parameters={
        #                                                   Parameters.PTProfileSimulation.INLETPRESSURE: p_wh,
        #                                                   Parameters.PTProfileSimulation.OUTLETPRESSURE: p_fl,
        #                                                   Parameters.PTProfileSimulation.FLOWRATETYPE: Constants.FlowRateType.LIQUIDFLOWRATE,
        #                                                   Parameters.PTProfileSimulation.CALCULATEDVARIABLE: Constants.CalculatedVariable.FLOWRATE},
        #                                               system_variables=system_variables,
        #                                               profile_variables=profile_variables)

        # global FLAG_CRITICAL
        # FLAG_CRITICAL = False
        # for f in range(len(results.messages)):
        #     if '*** CRITICAL FLOW CONDITIONS ***' in results.messages[f]:
        #         FLAG_CRITICAL = True
        #         break
        #
        # global EXTRA_DP
        # EXTRA_DP = 0
        # for g in range(len(results.cases)):
        #     if 'Extra DP' in results.cases[g]:
        #         ex_dp_pr = results.cases[g]
        #         ind_1 = ex_dp_pr.index('=', 0, len(ex_dp_pr)) + 1
        #         ind_2 = ex_dp_pr.index('psia', 0, len(ex_dp_pr))
        #         EXTRA_DP = float(ex_dp_pr[ind_1:ind_2]) / 0.000145038
        #         break
        #
        # global FLOWRATE
        # FLOWRATE = 0
        # for g in range(len(results.cases)):
        #     if 'Flowrate' in results.cases[g]:
        #         ex_dp_pr = results.cases[g]
        #         ind_1 = ex_dp_pr.index('=', 0, len(ex_dp_pr)) + 1
        #         ind_2 = ex_dp_pr.index('sbbl/day', 0, len(ex_dp_pr))
        #         FLOWRATE = float(ex_dp_pr[ind_1:ind_2]) * 0.158987294928
        #         break
        #
        # global SONIC_EC
        # global VEL_OUT_EC
        # global VEL_IN_EC
        #
        # global CD
        # global CSP
        # global CSPG
        # global CSPL
        #
        # global DP
        #
        # for f in range(len(results.messages)):
        #     if 'Velocity out' in results.messages[f]:
        #         sonic_eng_cnsl = results.messages[f]
        #         ind_1 = sonic_eng_cnsl.index('Sonic', 0, len(sonic_eng_cnsl))
        #         ind_2 = sonic_eng_cnsl.index('Mach', 0, len(sonic_eng_cnsl))
        #         sonic_eng_cnsl_ = sonic_eng_cnsl[ind_1:ind_2]
        #         ind_3 = sonic_eng_cnsl_.index('=', 0, len(sonic_eng_cnsl_))
        #         SONIC_EC = float(sonic_eng_cnsl_[ind_3 + 1:]) * 0.3048
        #
        #         ind_4 = sonic_eng_cnsl.index('Velocity out', 0, len(sonic_eng_cnsl))
        #         vel_out_cnsl_ = sonic_eng_cnsl[ind_4:ind_1]
        #         ind_5 = vel_out_cnsl_.index('=', 0, len(vel_out_cnsl_))
        #         VEL_OUT_EC = float(vel_out_cnsl_[ind_5 + 1:]) * 0.3048
        #         break
        #
        #     if 'Velocity in' in results.messages[f]:
        #         sonic_eng_cnsl_in = results.messages[f]
        #         ind_6 = sonic_eng_cnsl_in.index('Sonic', 0, len(sonic_eng_cnsl_in))
        #         ind_7 = sonic_eng_cnsl_in.index('Mach', 0, len(sonic_eng_cnsl_in))
        #         sonic_eng_cnsl_in_ = sonic_eng_cnsl_in[ind_6:ind_7]
        #         ind_8 = sonic_eng_cnsl_in_.index('=', 0, len(sonic_eng_cnsl_in_))
        #         SONIC_IN_EC = float(sonic_eng_cnsl_in_[ind_8 + 1:]) * 0.3048
        #
        #         ind_9 = sonic_eng_cnsl_in.index('Velocity in', 0, len(sonic_eng_cnsl_in))
        #         vel_in_cnsl_ = sonic_eng_cnsl_in[ind_9:ind_6]
        #         ind_10 = vel_in_cnsl_.index('=', 0, len(vel_in_cnsl_))
        #         VEL_IN_EC = float(vel_in_cnsl_[ind_10 + 1:]) * 0.3048
        #
        #     if 'Cd' in results.messages[f]:
        #         cd_eng_cnsl = results.messages[f]
        #         ind_11 = cd_eng_cnsl.index('Cd', 0, len(cd_eng_cnsl))
        #         ind_12 = cd_eng_cnsl.index('Csp', 0, len(cd_eng_cnsl))
        #         cd_eng_cnsl_ = cd_eng_cnsl[ind_11:ind_12]
        #         ind_13 = cd_eng_cnsl_.index('=', 0, len(cd_eng_cnsl_))
        #         CD = float(cd_eng_cnsl_[ind_13 + 1:])
        #
        #         csp_eng_cnsl_ = cd_eng_cnsl[ind_12:len(cd_eng_cnsl)]
        #         ind_14 = csp_eng_cnsl_.index('=', 0, len(csp_eng_cnsl_))
        #         CSP = float(csp_eng_cnsl_[ind_14 + 1:])
        #
        #     if 'Gas Csp' in results.messages[f]:
        #         cspg_eng_cnsl = results.messages[f]
        #         ind_15 = cspg_eng_cnsl.index('Gas Csp', 0, len(cspg_eng_cnsl))
        #         ind_16 = cspg_eng_cnsl.index('Liquid Csp', 0, len(cspg_eng_cnsl))
        #         cspg_eng_cnsl_ = cspg_eng_cnsl[ind_15:ind_16]
        #         ind_17 = cspg_eng_cnsl_.index('=', 0, len(cspg_eng_cnsl_))
        #         CSPG = float(cspg_eng_cnsl_[ind_17 + 1:])
        #
        #         cspl_eng_cnsl_ = cspg_eng_cnsl[ind_16:len(cspg_eng_cnsl)]
        #         ind_18 = cspl_eng_cnsl_.index('=', 0, len(cspl_eng_cnsl_))
        #         CSPL = float(cspl_eng_cnsl_[ind_18 + 1:])
        #
        #     if 'DeltaP' in results.messages[f]:
        #         dp_eng_cnsl = results.messages[f]
        #         ind_19 = dp_eng_cnsl.index('DeltaP', 0, len(dp_eng_cnsl))
        #         ind_20 = dp_eng_cnsl.index('DP Crit', 0, len(dp_eng_cnsl))
        #         dp_eng_cnsl_ = dp_eng_cnsl[ind_19:ind_20]
        #         ind_21 = dp_eng_cnsl_.index('=', 0, len(dp_eng_cnsl_))
        #         DP = float(dp_eng_cnsl_[ind_21 + 1:]) / 0.000145038

        global CHOKE_INDEX
        CHOKE_INDEX = results.profile[results.cases[0]]['BranchEquipment'].index('Choke 1')

        # profile results
        profile_results = {}
        for case, profile in results.profile.items():
            profile_df = pd.DataFrame.from_dict(profile)

        fluid_data_new = deepcopy(fluid_data)

        if 'use_table_model' in black_oil_model:
            if black_oil_model['use_table_model']:
                props_to_generate = [key for key in black_oil_model['table_model_data'].keys()
                                     if black_oil_model['table_model_data'][key] is None]
                if len(props_to_generate) > 0:
                    # табличная pvt для двумерного массива давленя и температур
                    p_max = 10000000000
                    p_min = 0
                    interp_func = {}
                    table_bo = []
                    table_rs = []
                    table_rho_wat = []
                    table_rho_oil = []
                    table_rho_gas = []
                    table_muw = []
                    table_muo = []
                    table_mug = []
                    table_pb = []
                    table_z = []
                    table_compro = []

                    t_before_choke = uc.convert_temperature(profile_df['Temperature'][CHOKE_INDEX - 1], 'C', 'K')
                    t_after_choke = uc.convert_temperature(profile_df['Temperature'][CHOKE_INDEX], 'C', 'K')
                    # t_array = np.arange(profile_df['Temperature'][CHOKE_INDEX - 1], profile_df['Temperature'][CHOKE_INDEX], step)
                    t_array = np.array([t_before_choke, t_after_choke])

                    for t_res_local in t_array:
                        # model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
                        # geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, 2],
                        #                      Parameters.GeothermalSurvey.TEMPERATURE: [float(t_res_local - 273.15),
                        #                                                                float(t_res_local - 273.15)]}
                        # geothermal_df = pd.DataFrame(geothermal_survey)
                        # model.set_geothermal_profile(Well=well_name, value=geothermal_df)
                        #
                        # results = model.tasks.ptprofilesimulation.run(producer=well_name,
                        #                                               parameters=parameters,
                        #                                               profile_variables=self.profile_variables)
                        # вытаскиваем результаты
                        table_p_1d = self.define_pvt_table_data(results, props_to_generate,
                                                                black_oil_model['table_model_data'])
                        # найдем min и max значение давления на всех рассчетах
                        p_max = min(table_p_1d['bo'].index.values[0], p_max)
                        p_min = max(table_p_1d['bo'].index.values[-1], p_min)
                        p_array = np.arange(p_max, p_min, -0.1)

                        interp_bo = interp1d(table_p_1d['bo'].index.values, table_p_1d['bo'].iloc[:, 0].values,
                                             kind='linear')
                        interp_rs = interp1d(table_p_1d['rs'].index.values, table_p_1d['rs'].iloc[:, 0].values,
                                             kind='linear')
                        interp_rho_wat = interp1d(table_p_1d['rw'].index.values,
                                                  table_p_1d['rw'].iloc[:, 0].values, kind='linear')
                        interp_rho_oil = interp1d(table_p_1d['ro'].index.values,
                                                  table_p_1d['ro'].iloc[:, 0].values, kind='linear')
                        interp_rho_gas = interp1d(table_p_1d['rg'].index.values,
                                                  table_p_1d['rg'].iloc[:, 0].values, kind='linear')
                        interp_muw = interp1d(table_p_1d['muw'].index.values,
                                              table_p_1d['muw'].iloc[:, 0].values, kind='linear')
                        interp_muo = interp1d(table_p_1d['muo'].index.values,
                                              table_p_1d['muo'].iloc[:, 0].values, kind='linear')
                        interp_mug = interp1d(table_p_1d['mug'].index.values,
                                              table_p_1d['mug'].iloc[:, 0].values, kind='linear')
                        interp_pb = interp1d(table_p_1d['pb'].index.values,
                                             table_p_1d['pb'].iloc[:, 0].values, kind='linear')
                        interp_z = interp1d(table_p_1d['z'].index.values,
                                            table_p_1d['z'].iloc[:, 0].values, kind='linear')
                        interp_compro = interp1d(table_p_1d['co'].index.values,
                                                 table_p_1d['co'].iloc[:, 0].values, kind='linear')
                        # сохраним занчения
                        interp_func[t_res_local] = {"bo": interp_bo, "rs": interp_rs, 'rw': interp_rho_wat,
                                                    'ro': interp_rho_oil, 'rg': interp_rho_gas,
                                                    'muw': interp_muw,
                                                    'muo': interp_muo, 'mug': interp_mug, 'pb': interp_pb,
                                                    'z': interp_z,
                                                    'co': interp_compro}
                    # интерполяция всех pvt
                    for t_res_local in t_array:
                        for key in interp_func[t_res_local]:
                            if key == 'bo':
                                col = interp_func[t_res_local][key](p_array)
                                table_bo.append(col)
                            if key == 'rs':
                                col = interp_func[t_res_local][key](p_array)
                                table_rs.append(col)
                            if key == 'rw':
                                col = interp_func[t_res_local][key](p_array)
                                table_rho_wat.append(col)
                            if key == 'rg':
                                col = interp_func[t_res_local][key](p_array)
                                table_rho_gas.append(col)
                            if key == 'ro':
                                col = interp_func[t_res_local][key](p_array)
                                table_rho_oil.append(col)
                            if key == 'muw':
                                col = interp_func[t_res_local][key](p_array)
                                table_muw.append(col)
                            if key == 'muo':
                                col = interp_func[t_res_local][key](p_array)
                                table_muo.append(col)
                            if key == 'mug':
                                col = interp_func[t_res_local][key](p_array)
                                table_mug.append(col)
                            if key == 'pb':
                                col = interp_func[t_res_local][key](p_array)
                                table_pb.append(col)
                            if key == 'z':
                                col = interp_func[t_res_local][key](p_array)
                                table_z.append(col)
                            if key == 'co':
                                col = interp_func[t_res_local][key](p_array)
                                table_compro.append(col)
                        # конец создания двумерной таблицы с pvt

                        # Сохраним в табличные данные
                    table_model_data_new = {"bo": pd.DataFrame(data=np.transpose(table_bo), columns=t_array,
                                                               index=p_array),
                                            "rs": pd.DataFrame(data=np.transpose(table_rs), columns=t_array,
                                                               index=p_array),
                                            "rw": pd.DataFrame(data=np.transpose(table_rho_wat),
                                                               columns=t_array,
                                                               index=p_array),
                                            "ro": pd.DataFrame(data=np.transpose(table_rho_oil),
                                                               columns=t_array,
                                                               index=p_array),
                                            "rg": pd.DataFrame(data=np.transpose(table_rho_gas),
                                                               columns=t_array,
                                                               index=p_array),
                                            "mug": pd.DataFrame(data=np.transpose(table_mug), columns=t_array,
                                                                index=p_array),
                                            "muo": pd.DataFrame(data=np.transpose(table_muo), columns=t_array,
                                                                index=p_array),
                                            "muw": pd.DataFrame(data=np.transpose(table_muw), columns=t_array,
                                                                index=p_array),
                                            "co": pd.DataFrame(data=np.transpose(table_compro),
                                                               columns=t_array,
                                                               index=p_array),
                                            'z': pd.DataFrame(data=np.transpose(table_z), columns=t_array,
                                                              index=p_array),
                                            'pb': pd.DataFrame(data=np.transpose(table_pb), columns=t_array,
                                                               index=p_array)}

                    fluid_data_new['pvt_model_data']['black_oil'].update({'table_model_data': table_model_data_new})
                    # model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
                    # geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, 2],
                    #                      Parameters.GeothermalSurvey.TEMPERATURE: [
                    #                          profile_df['Temperature'][CHOKE_INDEX - 1],
                    #                          profile_df['Temperature'][CHOKE_INDEX]]}
                    #
                    # geothermal_df = pd.DataFrame(geothermal_survey)
                    # model.set_geothermal_profile(Well=well_name, value=geothermal_df)

        model.save()
        model.close()

        dp_array_pipesim = {}
        dp_array_pipesim['p2'] = uc.convert_pressure(profile_df['Pressure'][CHOKE_INDEX - 1], 'bar', 'pa')
        dp_array_pipesim['t2'] = uc.convert_temperature(profile_df['Temperature'][CHOKE_INDEX], 'C', 'K')
        dp_array_pipesim['p1'] = uc.convert_pressure(profile_df['Pressure'][CHOKE_INDEX], 'bar', 'pa')
        dp_array_pipesim['t1'] = uc.convert_temperature(profile_df['Temperature'][CHOKE_INDEX - 1], 'C', 'K')

        # dp_array_pipesim['dt'] = uc.convert_temperature(profile_df['Temperature'][CHOKE_INDEX - 1], 'C', 'K') - \
        #                          uc.convert_temperature(profile_df['Temperature'][CHOKE_INDEX], 'C', 'K')
        # dp_array_pipesim['t_check'] = None
        # dp_array_uniflocpy[step_num, 4] = choke.
        # if choke.regime_type == 'critical' or choke.regime_type == 'supercritical':
        #     dp_array_uniflocpy[step_num, 5] = 1
        # else:
        #     dp_array_uniflocpy[step_num, 5] = 0
        # dp_array_uniflocpy[step_num, 6] = choke.fluid_velocity
        # dp_array_uniflocpy['p_crit'] = choke.
        # dp_array_pipesim['extra_dp'] = EXTRA_DP
        # dp_array_pipesim['wct'] = wct / 100
        # dp_array_pipesim['rp'] = fluid_data['pvt_model_data']['black_oil']['rp']
        # dp_array_pipesim['qliq'] = fluid_data['q_fluid']
        # dp_array_pipesim['d_choke'] = d_choke
        # dp_array_pipesim['cd'] = CD
        # dp_array_uniflocpy[step_num, 14] = CD
        # dp_array_pipesim['c_vg'] = CSP
        # dp_array_uniflocpy['c_v'] = choke.c_vg
        # dp_array_pipesim['c_vl'] = CSPL

        return dp_array_pipesim, dp_array_pipesim['t2']

    def main(self,
             p1,
             t1,
             choke_data,
             fluid_data,
             model_path,
             flow_direction,
             ):
        error_results = {}

        pipesim_results, t2_pip = self.calc_model_pipesim(p1=p1,
                                                          t1=t1,
                                                          model_path=model_path,
                                                          fluid_data=fluid_data,
                                                          choke_data=choke_data,
                                                          flow_direction=flow_direction)
        unifloc_results, t2_un = self.calc_model_uniflocpy(p1=p1,
                                                           t1=t1,
                                                           choke_data=choke_data,
                                                           fluid_data=fluid_data,
                                                           flow_direction=flow_direction)

        columns_names = ['p2', 't2', 'p1', 't1', 't2_pip', 't2_un', 't1 - t2_pip',
                         't1 - t2_un', 'gf', 'regime', 'ql', 'qg', 'status']
        j = 0

        for i in columns_names:
            if i == 't2_pip':
                error_results['t2_pip'] = t2_pip
            elif i == 't1':
                error_results['t1'] = t1
            elif i == 't2_un':
                error_results['t2_un'] = t2_un
            elif i == 't1 - t2_pip':
                error_results['t1 - t2_pip'] = t1 - t2_pip
            elif i == 't1 - t2_un':
                error_results['t1 - t2_un'] = t1 - t2_un
            elif i == 'gf':
                error_results['gf'] = unifloc_results['gf']
            elif i == 'regime':
                error_results['regime'] = unifloc_results['regime']
            elif i == 'ql':
                error_results['ql'] = unifloc_results['ql'] * 86400
            elif i == 'qg':
                error_results['qg'] = unifloc_results['qg'] * 86400
            elif i == 'status':
                if (t1 - t2_pip) < 0 and (t1 - t2_un) > 0 or (t1 - t2_pip) > 0 and (t1 - t2_un) < 0:
                    error_results['status'] = 1  # несовпадение
                else:
                    error_results['status'] = 0  # совпадение
            else:
                error_results[columns_names[j]] = ((abs(
                    unifloc_results[columns_names[j]] - pipesim_results[columns_names[j]])) /
                                                   pipesim_results[columns_names[j]]) * 100
            j += 1
        result_dict = {'pipesim_results': pipesim_results, 'unifloc_results': unifloc_results,
                       'error_results': error_results}

        return result_dict
