"""
Подкласс для автотестирования газлифта (well)

Позволяет как вызывать сравнительный расчет вручную, так и на рандомных данных в определенных диапазонах данных,
генерируемых с помощью латинского гиперкуба.

31/01/2022

@alexey_vodopyan
@erik_ovsepyan
"""

import pandas as pd
from typing import Tuple, Union, Any, Optional
from smt.sampling_methods import LHS
from sixgill.pipesim import Model
from sixgill.definitions import ModelComponents, Parameters, Constants, Units, ProfileVariables, SystemVariables
import numpy as np
from unifloc.well.gaslift_well_several_valves import GasLiftWellSeveralValves
from unifloc.well.gaslift_well import GasLiftWell
from unifloc.tools import units_converter as uc
from scipy.interpolate import interp1d
from copy import deepcopy
import math as mt
from NEW_autotest.autotest_class import Autotest


class AutotestGasLift(Autotest):

    def __init__(self, model_path,
                 file_path,
                 trajectory_data,
                 ambient_temperature_data,
                 fluid_data,
                 pipe_data,
                 equipment_data,
                 hydr_corr_type):

        super().__init__(model_path,
                         file_path,
                         trajectory_data,
                         ambient_temperature_data,
                         fluid_data,
                         pipe_data,
                         equipment_data,
                         hydr_corr_type)

    def formate_pipesim_results_valve(self, array, valves):
        x_curve = array['depth']['depth'].values
        y_curve = array['P_atma']['P_atma'].values
        z_curve = array['p_cas']['p_cas'].values
        results = {}
        for i in range(len(valves)):
            x_point = valves[i].h_mes
            y_interp_func = interp1d(x_curve, y_curve, kind="linear", fill_value="extrapolate")
            y_result = y_interp_func(x_point).item()
            z_interp_func = interp1d(x_curve, z_curve, kind="linear", fill_value="extrapolate")
            z_result = z_interp_func(x_point).item()
            p = y_curve[-1]
            results["P_atma"] = p
            results["h_mes" + str(i + 1)] = x_point
            results["d_port" + str(i + 1)] = valves[i].d
            results["s_bellow" + str(i + 1)] = valves[i].s_bellow
            results["p_valve" + str(i + 1)] = valves[i].p_valve / 101325
            results["p_cas" + str(i + 1)] = z_result / 101325
            results["p_tub" + str(i + 1)] = y_result / 101325
            results["R" + str(i + 1)] = valves[i].r
            results["p_dome" + str(i + 1)] = (valves[i].p_valve * (1 - valves[i].r)) / 101325
            results["p_close" + str(i + 1)] = (valves[i].p_valve * (1 - valves[i].r)) / 101325
            popen = (valves[i].p_valve * (1 - valves[i].r) - y_result * valves[i].r) / (1 - valves[i].r)
            results["p_open" + str(i + 1)] = popen / 101325
            if y_result < z_result and z_result > popen:
                results["status" + str(i + 1)] = "open"
            else:
                results["status" + str(i + 1)] = "closed"

        return results

    def __find_change_parameter(self, keys: list, data: list, pfl: float, freq_q_ag: float, model_path: str,
                                fluid_data: dict, pipe_data: dict, equipment_data: dict, p_gas_inj: float,
                                limited_pars: dict) -> \
            Tuple[Union[float, Any], Union[float, Any], str, dict, dict, Optional[dict]]:
        """
        Функция для обновления значений переменных соответственно значениям латинского гиперкуба

        Parameters
        ----------
        :param keys: названия параметров для которых проводим семплирование, list
        :param data: строка с одним набором параметров, list
        :param pfl: линейное давление, атма, float
        :param freq_q_ag: частота вращения ЭЦН или расход закачки газлифтного газа, Гц или ст. м3/с, float
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
        freq_q_ag_new = None
        fluid_data_new = None
        pipe_data_new = None
        equipment_data_new = None
        p_gas_inj_new = None
        init_dicts = [fluid_data, pipe_data, equipment_data]

        # Цикл поиска ключей в исходных данных
        for i in range(len(keys)):
            if keys[i] == 'pfl':
                pfl_new = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 'freq_q_ag':
                freq_q_ag_new = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 'p_gas_inj':
                p_gas_inj_new = data[i]
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
                            pipe_data_new = dic_new
                        else:
                            equipment_data_new = dic_new

                        model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))

                        # Заменим элемент, тк следующий ключ может попасться в этом же словаре
                        init_dicts[j] = dic_new

                        # Нет смысла продолжать поиск, тк каждый ключ уникален
                        break

        # Вернем путь в формат Pipesim
        model_path_new += '.pips'

        # Если данные не обновлялись, значит оставим старые
        if freq_q_ag_new is None:
            freq_q_ag_new = freq_q_ag

        if pfl_new is None:
            pfl_new = pfl

        if fluid_data_new is None:
            fluid_data_new = fluid_data

        if pipe_data_new is None:
            pipe_data_new = pipe_data

        if equipment_data_new is None:
            equipment_data_new = equipment_data

        if p_gas_inj_new is None:
            p_gas_inj_new = p_gas_inj

        return pfl_new, freq_q_ag_new, model_path_new, fluid_data_new, pipe_data_new, equipment_data_new, p_gas_inj_new

    def sample_model(self,
                     pars_limits: dict,
                     number_of_samples: int,
                     pfl: float,
                     model_path: str,
                     fluid_data: dict,
                     pipe_data: dict,
                     calc_options: dict,
                     limited_pars: dict = None,
                     freq_q_ag: float = None,
                     equipment_data: dict = None,
                     well_trajectory_data: dict = None,
                     calc_type: str = 'well',
                     result_path: str = 'results.xlsx',
                     temperature_option: str = "CONST",
                     heat_balance: bool = False,
                     ambient_temperature_data=None,
                     p_gas_inj=None
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

        Returns
        -------

        """

        # pfl = uc.convert_pressure(pfl, 'pa', 'atm')
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

        results_df_ps = pd.DataFrame(columns=keys + ['Error', 'Density_inversion_flag'],
                                     index=range(number_of_samples))

        results_df_up = pd.DataFrame(columns=keys + ['Error', 'Density_inversion_flag'],
                                     index=range(number_of_samples))

        results_df_err = pd.DataFrame(columns=keys + ['Error', 'Density_inversion_flag'],
                                      index=range(number_of_samples))

        # Итерируемся по набору данных и считаем модели
        for i in range(number_of_samples):

            print(f'Расчет {i + 1} из {number_of_samples}...')

            # Определим, что за параметры и изменим их значения
            pfl_new, freq_q_ag_new, model_path_new, fluid_data_new, pipe_data_new, equipment_data_new, p_gas_inj_new = \
                self.__find_change_parameter(keys,
                                             data[i, :],
                                             pfl,
                                             freq_q_ag,
                                             model_path,
                                             fluid_data,
                                             pipe_data,
                                             equipment_data,
                                             p_gas_inj,
                                             limited_pars)

            # Передадим в pipesim и проведем расчет
            try:
                results_dict = self.main(well_trajectory_data=well_trajectory_data,
                                         fluid_data=fluid_data_new,
                                         pipe_data=pipe_data_new,
                                         calc_type=calc_type,
                                         equipment_data=equipment_data_new,
                                         freq=freq_q_ag_new,
                                         pfl=pfl_new,
                                         model_path=model_path_new,
                                         calc_options=calc_options,
                                         temperature_option=temperature_option,
                                         heat_balance=heat_balance,
                                         ambient_temperature_data=ambient_temperature_data,
                                         p_gas_inj=p_gas_inj_new,
                                         result_path=result_path)

            except (IndexError, KeyError):
                continue

            if np.isnan(results_dict['error_results']['P_atma']):
                continue

            results_df_ps.loc[i, keys] = data[i, :]
            results_df_up.loc[i, keys] = data[i, :]
            results_df_err.loc[i, keys] = data[i, :]

            for key in results_dict['uniflocpy_results']:
                results_df_up.loc[i, key] = results_dict['uniflocpy_results'][key]

            for key in results_dict['pipesim_results']:
                results_df_ps.loc[i, key] = results_dict['pipesim_results'][key]

            for key in results_dict['error_results']:
                results_df_err.loc[i, key] = results_dict['error_results'][key]

        results_df_ps.dropna(how='all', inplace=True)
        results_df_up.dropna(how='all', inplace=True)
        results_df_err.dropna(how='all', inplace=True)

        if calc_options['scenario']:
            return results_df_err

        with pd.ExcelWriter(result_path) as writer:
            results_df_ps.to_excel(writer, sheet_name='Pipesim')
            results_df_up.to_excel(writer, sheet_name='UniflocPy')
            results_df_err.to_excel(writer, sheet_name='Error')

        return

    def calc_model_uniflocpy(self,
                             fluid_data: dict,
                             pipe_data: dict, pfl: float = None,
                             freq_q_ag: float = None,
                             p_gas_inj: float = None,
                             equipment_data: dict = None,
                             well_trajectory_data: dict = None,
                             ambient_temperature_data: dict = None,
                             profile_variables: list = None,
                             calculation_type: str = 'well',
                             calc_options: dict = None):

        if equipment_data is None:
            equipment_data = {}

        if profile_variables is None:
            profile_variables = self.profile_variables

        q_liq = fluid_data['q_fluid']
        wct = fluid_data['wct']

        if pipe_data['tubing']['bottom_depth'] == pipe_data['casing']['bottom_depth'] or \
                pipe_data['tubing']['bottom_depth'] > pipe_data['casing']['bottom_depth']:
            raise ValueError('Для расчета газлифта(well), длина НКТ не должна быть равна '
                             'или быть больше длины обсадной колонны')
        if pipe_data['tubing']['d'] > pipe_data['casing']['d']:
            raise ValueError('Для расчета газлифта(well), диаметр НКТ не должен превышать диаметр обсадной колонны'
                             'Поменяйте граничные условия в pars_limits, либо увеличьте диаметр обсадной')

        if calculation_type.lower() == 'well':
            # Расчет целой скважины
            qinj = None

            if 'gl_system' in equipment_data:
                # Скважина с Газлифтом
                well = GasLiftWellSeveralValves(fluid_data=fluid_data,
                                                pipe_data=pipe_data,
                                                equipment_data=equipment_data,
                                                well_trajectory_data=well_trajectory_data,
                                                ambient_temperature_data=ambient_temperature_data)

                result_gl = well.calc_pwf_pfl(p_fl=pfl,
                                              q_liq=q_liq,
                                              wct=wct,
                                              q_gas_inj=freq_q_ag,
                                              p_gas_inj=p_gas_inj,
                                              step_length=self.L_REPORT,
                                              output_params=True,
                                              hydr_corr_type=self.hydr_corr_type)

                qinj = freq_q_ag
                status_error = ""

                results = {}
                for i in range(len(well.gl_system.valves)):
                    p_up = result_gl[2]
                    results["P_atma"] = p_up
                    results["h_mes" + str(i + 1)] = well.gl_system.valves[i].h_mes
                    results["d_port" + str(i + 1)] = well.gl_system.valves[i].d
                    results["s_bellow" + str(i + 1)] = well.gl_system.valves[i].s_bellow
                    results["p_valve" + str(i + 1)] = well.gl_system.valves[i].p_valve / 101325
                    results["p_cas" + str(i + 1)] = well.gl_system.valves[i].p_cas / 101325
                    results["p_tub" + str(i + 1)] = well.gl_system.valves[i].p_tub / 101325
                    results["R" + str(i + 1)] = well.gl_system.valves[i].r
                    results["p_dome" + str(i + 1)] = well.gl_system.valves[i].p_dome / 101325
                    results["p_close" + str(i + 1)] = well.gl_system.valves[i].p_close / 101325
                    results["p_open" + str(i + 1)] = well.gl_system.valves[i].p_open / 101325
                    if well.gl_system.valves[i].h_mes == well.gl_system.valve_working.h_mes:
                        results["status" + str(i + 1)] = well.gl_system.valves[i].status + "_work"
                    else:
                        results["status" + str(i + 1)] = well.gl_system.valves[i].status

            else:
                # Фонтанная скважина
                print('Тип скважины - фонтан')
                well = GasLiftWell(fluid_data=fluid_data,
                                   pipe_data=pipe_data,
                                   equipment_data=equipment_data,
                                   well_trajectory_data=well_trajectory_data,
                                   ambient_temperature_data=ambient_temperature_data)

                well.calc_pwf_pfl(p_fl=pfl,
                                  q_liq=q_liq,
                                  wct=wct,
                                  step_length=self.L_REPORT)

            # if calc_options['save_results']:
            depth_list = [i * (-1) for i in well.extra_output['depth']]
            new_ann_distr = well.extra_output_annulus['p']
            for i in range(len(well.extra_output_annulus['p']), len(well.extra_output['p'])):
                new_ann_distr.append(0)

            results_distr = {'P_atma': pd.DataFrame(index=depth_list,
                                                    data=np.array(well.extra_output['p'])),
                             't': pd.DataFrame(index=depth_list,
                                               data=np.array(well.extra_output['t'])),
                             'p_cas': pd.DataFrame(index=depth_list,
                                                   data=np.array(new_ann_distr)),
                             'depth': pd.DataFrame(index=depth_list,
                                                   data=depth_list)
                             }

            return results, results_distr, qinj, well.gl_system.valves

    def calc_model_pipesim(self,
                           pfl: float,
                           model_path: str,
                           fluid_data: dict,
                           pipe_data: dict,
                           freq_q_ag: float = None,
                           equipment_data: dict = None,
                           well_trajectory_data: dict = None,
                           heat_balance=None,
                           temperature_option: str = 'CONST',
                           profile_variables: list = None,
                           ambient_temperature_data=None,
                           p_gas_inj=None,
                           calc_options=None):

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
        s_wall_cas = pipe_data['casing']['s_wall'] * 1000

        h_tub = pipe_data['tubing']['bottom_depth']
        d_tub = pipe_data['tubing']['d'] * 1000
        roughness_tub = pipe_data['tubing']['roughness'] * 1000
        s_wall_tub = pipe_data['tubing']['s_wall'] * 1000

        system_variables = [
            SystemVariables.PRESSURE,
            SystemVariables.TEMPERATURE,
            SystemVariables.PUMP_POWER,
            SystemVariables.POWER,
            SystemVariables.ESP_POWER,
            SystemVariables.BOTTOM_HOLE_PRESSURE,
            SystemVariables.ESP_INTAKE_GAS_VOLUME_FRACTION,
            SystemVariables.ESP_INTAKE_PRESSURE
        ]

        model = Model.new(model_path, units=Units.METRIC, overwrite=True)
        model.save()
        model.close()

        model = Model.open(model_path, units=Units.METRIC)

        self.pipesim_model(model=model,
                           modelcomponents=ModelComponents,
                           flow_direction=None,
                           h_start=None,
                           well_name=well_name,
                           temperature_option=temperature_option,
                           well_trajectory_data=well_trajectory_data,
                           h_cas=h_cas,
                           d_cas=d_cas,
                           roughness_cas=roughness_cas,
                           s_wall_cas=s_wall_cas,
                           h_tub=h_tub,
                           d_tub=d_tub,
                           roughness_tub=roughness_tub,
                           s_wall_tub=s_wall_tub,
                           gor=gor,
                           wct=wct,
                           dod=dod,
                           gamma_water=gamma_water,
                           gamma_gas=gamma_gas,
                           equipment_data=equipment_data,
                           freq_q_ag=freq_q_ag,
                           t_res=t_res,
                           p_gas_inj=p_gas_inj)

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
                                             hydr_corr_type=self.hydr_corr_type,
                                             p=pfl,
                                             qliq=qliq,
                                             h_tub=h_tub,
                                             modelcomponents=ModelComponents)

        # Проверка необходима ли генерация таблиц table_model_data
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

        model.add(ModelComponents.PACKER, 'Packer 1', context=well_name,
                  parameters={Parameters.Packer.TOPMEASUREDDEPTH: h_tub - 1})

        if 'gl_system' in equipment_data:
            h_valve = {}
            i_ = 0
            for valve in equipment_data['gl_system']:
                h_valve[valve] = equipment_data['gl_system'][valve]['h_mes']
                model.add(ModelComponents.GASLIFTINJECTION, "Inj" + str(i_), context=well_name,
                          parameters={Parameters.GasLiftInjection.TOPMEASUREDDEPTH: h_valve[valve],
                                      Parameters.GasLiftInjection.MANUFACTURER: "Weatherford",
                                      Parameters.GasLiftInjection.SERIES: "R-1",
                                      Parameters.GasLiftInjection.VALVETYPE: "IPO",
                                      Parameters.GasLiftInjection.VALVESIZE: 25.4,
                                      Parameters.GasLiftInjection.PORTSIZE: equipment_data['gl_system'][valve][
                                                                                'd'] * 1000,
                                      Parameters.GasLiftInjection.PORTAREA: ((mt.pi *
                                                                              equipment_data['gl_system'][valve][
                                                                                  'd'] ** 2) / 4) * 1000000,
                                      Parameters.GasLiftInjection.BELLOWAREA: equipment_data['gl_system'][valve][
                                                                                  's_bellow'] * 1000000,
                                      Parameters.GasLiftInjection.PTRO: uc.convert_pressure(
                                          equipment_data['gl_system'][valve]['p_valve'], "pa", "bar"),
                                      Parameters.GasLiftInjection.DISCHARGECOEFFICIENT: 1,
                                      Parameters.GasLiftInjection.DISCHARGETOFULLYOPEN: 116.35
                                      })
                i_ += 1

            model.save()

        results = model.tasks.ptprofilesimulation.run(producer=well_name,
                                                      parameters=parameters[0],
                                                      system_variables=system_variables,
                                                      profile_variables=profile_variables)

        model.save()
        model.close()

        global PRESSURE_INDEX
        self.PRESSURE_INDEX = uc.convert_pressure(results.profile[results.cases[0]]['Pressure'], 'bar', 'atm')[3:]

        global TEMPERATURE_INDEX
        self.TEMPERATURE_INDEX = uc.convert_temperature(results.profile[results.cases[0]]['Temperature'], 'C', 'K')[3:]

        global DEPTH
        self.DEPTH = results.profile[results.cases[0]][ProfileVariables.MEASURED_DEPTH][:1:-1]

        flag = self.analyze_density_inversion(results)

        return results, fluid_data_new, flag

    def __calc_error_gaslift(self, uniflocpy_value, pipesim_value):
        """

        Parameters
        ----------
        :param uniflocpy_array: массив uniflocpy
        :param pipesim_array: массив pipesim

        Returns
        -------

        """

        if type(uniflocpy_value) != str and type(pipesim_value) != str:
            error = (abs(uniflocpy_value - pipesim_value) / pipesim_value) * 100
        else:
            if uniflocpy_value == 'open' or uniflocpy_value == 'open_work' and pipesim_value == 'open':
                error = 'Совпадение - открыты'
            elif uniflocpy_value == 'closed' or uniflocpy_value == 'closed_work' and pipesim_value == 'closed':
                error = 'Совпадение - закрыты'
            else:
                error = 'Не совпало'

        return error

    def calc_error(self, uniflocpy_results=None, pipesim_results=None) -> dict:
        """
        Функция для расчета ошибки

        Parameters
        ----------
        :param uniflocpy_results : результаты из UniflocPy
        :param pipesim_results : результаты из Pipesim

        Returns
        -------
        словарь ошибок, dict
        """

        # Cловарь средних ошибок
        error_dict = {}

        if uniflocpy_results is not None and pipesim_results is not None:
            # По идее гарантировано, что ключи в словарях одинаковые
            for key in pipesim_results.keys():
                unifloc_key = uniflocpy_results[key]
                pipesim_key = pipesim_results[key]
                error = self.__calc_error_gaslift(unifloc_key, pipesim_key)

                # Сохраним в словарь средних ошибок
                error_dict.update({key: [error]})
        else:
            print('Один из массивов не заполнен, невозможно рассчитать ошибку')
        return error_dict

    def main(self,
             well_trajectory_data: object,
             fluid_data: object,
             pipe_data: object,
             calc_type: object,
             equipment_data: object,
             freq: object,
             pfl: object,
             model_path: object,
             result_path: str = None,
             calc_options: object = None,
             heat_balance: object = None,
             temperature_option: object = None,
             ambient_temperature_data: object = None,
             p_gas_inj: object = None) -> object:

        if calc_options is None:
            calc_options = {'error_calc': True,
                            'save_results': True,
                            'plot_results': True}

        if equipment_data is not None:
            equipment_data_pipesim = equipment_data

        uniflocpy_results, uniflocpy_results_distr, qinj_uniflocpy, status_error = \
            self.calc_model_uniflocpy(pfl=uc.convert_pressure(pfl, 'atm', 'pa'),
                                      fluid_data=fluid_data,
                                      freq_q_ag=freq,
                                      pipe_data=pipe_data,
                                      equipment_data=equipment_data,
                                      well_trajectory_data=well_trajectory_data,
                                      calculation_type=calc_type,
                                      profile_variables=self.profile_variables,
                                      ambient_temperature_data=ambient_temperature_data,
                                      p_gas_inj=p_gas_inj,
                                      calc_options=calc_options)
        # Запуск расчета на Pipesim
        pipesim_results, fluid_data_new, flag = self.calc_model_pipesim(pfl=uc.convert_pressure(pfl, 'atm', 'pa'),
                                                                        model_path=model_path,
                                                                        fluid_data=fluid_data,
                                                                        pipe_data=pipe_data,
                                                                        heat_balance=heat_balance,
                                                                        equipment_data=equipment_data_pipesim,
                                                                        well_trajectory_data=well_trajectory_data,
                                                                        freq_q_ag=qinj_uniflocpy,
                                                                        temperature_option=temperature_option,
                                                                        ambient_temperature_data=ambient_temperature_data,
                                                                        p_gas_inj=p_gas_inj,
                                                                        calc_options=calc_options)

        # Приведем результаты Pipesim к единому с UniflocPy формату

        pipesim_results_f = self.formate_pipesim_results(pipesim_results,
                                                         calc_type,
                                                         list(uniflocpy_results.keys()),
                                                         equipment_data)

        pipesim_results_valves = self.formate_pipesim_results_valve(pipesim_results_f, status_error)

        if calc_options['error_calc']:
            # Запустим расчет ошибки
            error_results = self.calc_error(uniflocpy_results, pipesim_results_valves)
        else:
            error_results = None

        if calc_options['save_results']:
            self.save_results(uniflocpy_results_distr, pipesim_results_f, error_results, result_path, calc_type,
                              equipment_data)

        if calc_options['plot_results']:
            self.plot_results(uniflocpy_results_distr, pipesim_results_f, calc_type)

        return {'pipesim_results': pipesim_results_valves, 'uniflocpy_results': uniflocpy_results,
                'error_results': error_results}
