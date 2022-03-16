"""
Подкласс для автотестирования ЭЦН (esp) и скважины с ЭЦН (well)

Позволяет как вызывать сравнительный расчет вручную, так и на рандомных данных в определенных диапазонах данных,
генерируемых с помощью латинского гиперкуба.

31/01/2022

@alexey_vodopyan
@erik_ovsepyan
"""

import pandas as pd
from smt.sampling_methods import LHS
from sixgill.pipesim import Model
from sixgill.definitions import ModelComponents, Parameters, Constants, Units, ProfileVariables
import copy
import numpy as np
from unifloc.pvt.fluid_flow import FluidFlow
from unifloc.well.esp_well import EspWell
from unifloc.equipment.esp import Esp
from copy import deepcopy
from unifloc.tools import units_converter as uc
from NEW_autotest.autotest_class import Autotest


class AutotestESP(Autotest):

    def __init__(self,
                 model_path,
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

    def sample_model(self,
                     pars_limits: dict,
                     number_of_samples: int,
                     pfl: float,
                     model_path: str,
                     fluid_data: dict,
                     pipe_data: dict,
                     p_ann: float,
                     esp_data: pd.DataFrame,
                     calc_options: dict,
                     esp_id: int = None,
                     limited_pars: dict = None,
                     freq_q_ag: float = None,
                     stage_by_stage: bool = False,
                     equipment_data: dict = None,
                     well_trajectory_data: dict = None,
                     temperature_option: str = None,
                     calc_type: str = 'well',
                     result_path: str = 'results.xlsx',
                     heat_balance: bool = False,
                     flow_direction=None,
                     h_start=None,
                     mean_integral_error=None,
                     ambient_temperature_data=None
                     ):
        """
        Функция для расчета моделей на произвольном наборе параметров

        Parameters
        ----------
        :param pars_limits: словарь с параметром и его границами, в которых будет генерироваться данные, dict
        :param limited_pars: словарь с однозначными соответствиями параметров для объектов, которые могут повторяться, dict
            Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing
        :param number_of_samples: количество наборов параметров, integer
        :param pfl: линейное давление, атма, float
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

        keys = list(pars_limits.keys())
        xlimits = [pars_limits[par] for par in keys]

        # Создание латинского гиперкуба
        if isinstance(xlimits, list):
            xlimits = np.array(xlimits)
        sampling = LHS(xlimits=xlimits)

        # Генерация выбранного набора данных
        data = sampling(number_of_samples)

        # Проверка на stages, которые могут быть только целыми
        if 'stages' in keys:
            data[:, keys.index('stages')] = data[:, keys.index('stages')].astype(np.int)

        # Результирующая таблица
        results_df = pd.DataFrame(columns=keys + ['Error', 'Density_inversion_flag'],
                                  index=range(number_of_samples))

        # Для esp удобней смотреть дебит ГЖС, придется делать костыль :((
        if calc_type == 'esp':
            results_df.insert(len(results_df.columns) - 1, 'q_mix', np.NaN)

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
                                           limited_pars,
                                           freq_q_ag)

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
                                         mean_integral_error=mean_integral_error,
                                         stage_by_stage=stage_by_stage,
                                         esp_id=esp_id,
                                         esp_data=esp_data,
                                         freq=freq_q_ag_new,
                                         p_ann=p_ann)
            except IndexError:
                continue

            # if not calc_options['scenario']:
                # Сохраним значения аргументов
            results_df.loc[i, keys] = data[i, :]
            results_df.loc[i, 'Density_inversion_flag'] = results_dict['density_inversion']

            # if calc_type == 'esp':
            #     results_df.loc[i, 'q_mix'] = results_dict['uniflocpy_results']['q_mix']

            # Сохраним результаты
            results_df.loc[i, 'Error'] = [results_dict['error_results']]

        # Сконвертируем дебит в м3/сут
        if 'q_fluid' in results_df.columns:
            results_df['q_fluid'] = uc.convert_rate(results_df['q_fluid'], 'm3/s', 'm3/day')

        # Приведем результаты к удобному для вывода формату
        results_df.dropna(how='all', inplace=True)

        # if calc_options['scenario']:
        #     # scenario = pd.DataFrame(куы)
        #     return results_df

        if len(results_df) > 0:
            results_df = self.reformat_results(results_df, calc_type)
            if calc_options['scenario']:
                return results_df
            results_df.to_excel(result_path)

        return results_df

    def calc_model_uniflocpy(self,
                             ambient_temperature_data: dict,
                             fluid_data: dict,
                             pipe_data: dict,
                             pfl: float = None,
                             equipment_data: dict = None,
                             well_trajectory_data: dict = None,
                             calculation_type: str = 'well',
                             freq_q_ag: float = None,
                             p_ann: float = None):

        if equipment_data is None:
            equipment_data = {}

        q_liq = fluid_data['q_fluid']
        wct = fluid_data['wct']
        t_res = ambient_temperature_data["T"][1]
        rp = fluid_data["pvt_model_data"]["black_oil"]['rp']

        if calculation_type.lower() == 'well':  # Скважина с ЭЦН
            if 'esp_system' in equipment_data:
                well = EspWell(fluid_data=fluid_data,
                               pipe_data=pipe_data,
                               equipment_data=equipment_data,
                               well_trajectory_data=well_trajectory_data,
                               ambient_temperature_data=ambient_temperature_data)

                well.calc_pwf_pfl(p_fl=pfl,
                                  q_liq=q_liq,
                                  p_ann=p_ann,
                                  wct=wct,
                                  freq=freq_q_ag,
                                  step_length=self.L_REPORT)

                depth_list = [i * (-1) for i in well.extra_output['depth']]

                return {'P_atma': pd.DataFrame(index=depth_list,
                                               data=np.array(well.extra_output['p']) / 101325)}

        elif calculation_type.lower() == 'esp':  # ЭЦН

            # Обновление дополнительных параметров для инициализации Esp
            pvt = FluidFlow(**fluid_data)
            equipment_data['esp_system']['esp'].update({'fluid': pvt})
            equipment_data['esp_system']['esp'].update({'h_mes': pipe_data['tubing']['bottom_depth']})

            # Инициализация Esp

            esp = Esp(h_mes=pipe_data['tubing']['bottom_depth'],
                      stages=equipment_data['esp_system']['esp']['stages'],
                      esp_data=equipment_data['esp_system']['esp']['esp_data'],
                      fluid=pvt,
                      viscosity_correction=equipment_data['esp_system']['esp']['viscosity_correction'],
                      gas_correction=equipment_data['esp_system']['esp']['gas_correction'],
                      gas_degr_value=equipment_data['esp_system']['esp']['gas_degr_value'])

            # Найдем давление на приеме в Pipesim
            p_in = self.P_IN * 101325

            # Расчет эцн
            p = esp.calc_pt(q_liq=q_liq,
                            wct=wct,
                            p=p_in,
                            t=t_res,
                            rp=rp,
                            freq=freq_q_ag,
                            direction_to='dis',
                            extra_output=True)

            return {'esp_dp': [esp.dp], 'esp_power': [esp.power_esp], 'esp_head': [esp.head],
                    'esp_p_dis': [p[0]]}

    def calc_model_pipesim(self,
                           pfl: float,
                           model_path: str,
                           fluid_data: dict,
                           pipe_data: dict,
                           equipment_data: dict = None,
                           well_trajectory_data: dict = None,
                           heat_balance: bool = None,
                           temperature_option: str = 'CONST',
                           profile_variables: list = None,
                           freq_q_ag: float = None,
                           flow_direction=None,
                           h_start=None,
                           ambient_temperature_data=None,
                           calc_type=None
                           ):

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
        s_wall_cas = pipe_data['tubing']['s_wall']

        h_tub = pipe_data['tubing']['bottom_depth']
        d_tub = pipe_data['tubing']['d'] * 1000
        roughness_tub = pipe_data['tubing']['roughness'] * 1000
        s_wall_tub = pipe_data['casing']['s_wall']

        # Создадим модель, сохраним и закроем
        model = Model.new(model_path, units=Units.METRIC, overwrite=True)
        model.save()
        model.close()

        # Откроем модель снова, чтобы были метрические единицы измерения
        model = Model.open(model_path, units=Units.METRIC)

        self.pipesim_model(model=model,
                           modelcomponents=ModelComponents,
                           flow_direction=flow_direction,
                           h_start=h_start,
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
                           equipment_data=equipment_data)

        pip_parameters = self.pipesim_parameters(wct=wct,
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

        if black_oil_model['use_table_model']:
            self.use_table_model(black_oil_model=black_oil_model,
                                 h_cas=h_cas,
                                 parameters=pip_parameters[0],
                                 model=model,
                                 ambient_temperature_data=ambient_temperature_data,
                                 heat_balance_pipesim=pip_parameters[1],
                                 well_name=well_name,
                                 h_tub=h_tub,
                                 pfl=pfl,
                                 t_res=t_res,
                                 fluid_data_new=fluid_data_new)

        # Добавим насос в скважину, если он есть
        if 'esp_system' in equipment_data:

            esp_manufacture = equipment_data['esp_system']['esp']['esp_manufacture']
            esp_model = str(equipment_data['esp_system']['esp']['esp_id'])
            freq = freq_q_ag
            stages = equipment_data['esp_system']['esp']['stages']

            model.add(ModelComponents.ESP, "Esp 1", context=well_name,
                      parameters={Parameters.ESP.TOPMEASUREDDEPTH: h_tub,
                                  Parameters.ESP.OPERATINGFREQUENCY: freq,
                                  Parameters.ESP.MANUFACTURER: esp_manufacture,
                                  Parameters.ESP.MODEL: esp_model})

            model.set_value(Well=well_name, parameter=Parameters.ESP.NUMBERSTAGES, value=stages)

            if 'viscosity_correction' in equipment_data['esp_system']['esp']:
                viscosity_correction = equipment_data['esp_system']['esp']['viscosity_correction']

                if viscosity_correction:
                    model.set_value(context='Esp 1', parameter=Parameters.ESP.USEVISCOSITYCORRECTION,
                                    value=viscosity_correction)

            if 'stage_by_stage' in equipment_data['esp_system']['esp']:
                stage_by_stage = equipment_data['esp_system']['esp']['stage_by_stage']

                if stage_by_stage:
                    model.set_value(context='Esp 1', parameter=Parameters.ESP.USESTAGEBYSTAGECALC,
                                    value=stage_by_stage)
            if 'gas_correction' in equipment_data['esp_system']['esp']:
                gas_correction = equipment_data['esp_system']['esp']['gas_correction']

                if gas_correction:
                    gas_separate_coeff = equipment_data['esp_system']['esp']['gas_degr_value']

                    if gas_separate_coeff <= 2:
                        head_factor = gas_separate_coeff

                        model.set_value(context='Esp 1', parameter=Parameters.ESP.HEADFACTOR,
                                        value=head_factor)
        results = model.tasks.ptprofilesimulation.run(producer=well_name,
                                                      parameters=pip_parameters[0],
                                                      profile_variables=self.profile_variables)
        if calc_type == 'esp':
            p_res = results.profile[results.cases[0]]['Pressure'][0]
            new_parameters = {Parameters.PTProfileSimulation.INLETPRESSURE: p_res,
                              Parameters.PTProfileSimulation.LIQUIDFLOWRATE: qliq,
                              Parameters.PTProfileSimulation.FLOWRATETYPE: Constants.FlowRateType.LIQUIDFLOWRATE,
                              Parameters.PTProfileSimulation.CALCULATEDVARIABLE:
                                  Constants.CalculatedVariable.OUTLETPRESSURE}

            results = model.tasks.ptprofilesimulation.run(producer=well_name,
                                                          parameters=new_parameters,
                                                          profile_variables=self.profile_variables)

            p_in_index = results.profile[results.cases[0]]['BranchEquipment'].index('Esp 1') - 1

            # global Q_MIX
            # self.Q_MIX = results.profile[results.cases[0]][ProfileVariables.VOLUME_FLOWRATE_FLUID_INSITU][p_in_index]

            global P_IN
            self.P_IN = uc.convert_pressure(results.profile[results.cases[0]]['Pressure'][p_in_index], 'bar', 'atm')

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
             freq,
             esp_data,
             esp_id,
             pfl,
             model_path,
             p_ann,
             stage_by_stage=False,
             calc_options=None,
             result_path=None,
             heat_balance=None,
             temperature_option=None,
             flow_direction=None,
             h_start=None,
             mean_integral_error=None,
             ambient_temperature_data: object = None
             ) -> dict:

        """
        :param freq: частота ЭЦН, Гц
        :param esp_data: паспортные данные ЭЦН из базы насосов - pd.Series или dict
        :param esp_id: id ЭЦН
        :param p_ann: затрубное давление, атм
        :param stage_by_stage: запуск расчета от ступени к ступени
        :return:
        """
        if equipment_data is not None:
            equipment_data_pipesim = copy.deepcopy(equipment_data)
            if 'esp_system' in equipment_data:
                if 'esp' in equipment_data['esp_system']:
                    equipment_data['esp_system']['esp'].update({'esp_data': esp_data[esp_id]})
                    equipment_data_pipesim['esp_system']['esp'].update({'esp_manufacture': 'Unifloc',
                                                                        'esp_id': str(esp_id)})
                    if stage_by_stage:
                        equipment_data_pipesim['esp_system']['esp'].update({'stage_by_stage': stage_by_stage})
                elif calc_type == 'esp':
                    print('calc_type = "esp", но не заданы параметры насоса. Задайте параметры насоса')

            elif calc_type == 'esp':
                print('calc_type = "esp", но не заданы параметры насоса. Задайте параметры насоса')

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
                                                                        freq_q_ag=freq,
                                                                        calc_type=calc_type,
                                                                        temperature_option=temperature_option,
                                                                        ambient_temperature_data=ambient_temperature_data,
                                                                        flow_direction=flow_direction,
                                                                        h_start=h_start)

        # Запуск расчета на UniflocPy
        uniflocpy_results = \
            self.calc_model_uniflocpy(pfl=uc.convert_pressure(pfl, 'atm', 'pa'),
                                      fluid_data=fluid_data_new,
                                      freq_q_ag=freq,
                                      pipe_data=pipe_data,
                                      equipment_data=equipment_data,
                                      well_trajectory_data=well_trajectory_data,
                                      calculation_type=calc_type,
                                      ambient_temperature_data=ambient_temperature_data,
                                      p_ann=p_ann)

        # Приведем результаты Pipesim к единому с UniflocPy формату
        pipesim_results_f = self.formate_pipesim_results(pipesim_results, calc_type, list(uniflocpy_results.keys()),
                                                         equipment_data)

        if calc_options['error_calc']:
            # Запустим расчет ошибки
            error_results = self.calc_error(uniflocpy_results, pipesim_results_f, mean_integral_error)
        else:
            error_results = None

        if calc_options['save_results']:
            self.save_results(uniflocpy_results, pipesim_results_f, error_results, result_path, calc_type, equipment_data)

        if calc_options['plot_results']:
            self.plot_results(uniflocpy_results, pipesim_results_f, calc_type)

        # if calc_options['scenario']:
        #     return error_results
        return {'pipesim_results': pipesim_results_f, 'uniflocpy_results': uniflocpy_results,
                'error_results': error_results, 'density_inversion': flag}
