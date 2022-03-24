"""
Подкласс для автотестирования трубы (pipe).

Позволяет как вызывать сравнительный расчет вручную, так и на рандомных данных в определенных диапазонах данных,
генерируемых с помощью латинского гиперкуба.

31/01/2022

@alexey_vodopyan
@erik_ovsepyan
"""

import pandas as pd
from smt.sampling_methods import LHS
from sixgill.pipesim import Model
from sixgill.definitions import ModelComponents, Constants, Units, ProfileVariables
import copy
import numpy as np
from unifloc.common.trajectory import Trajectory
from unifloc.pvt.fluid_flow import FluidFlow
from unifloc.pipe.pipeline import Pipeline
from copy import deepcopy
from unifloc.tools import units_converter as uc
from NEW_autotest.autotest_class import Autotest

class AutotestPipe(Autotest):

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
                     calc_options: dict,
                     freq_q_ag = None,
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
        :param calc_options: словарь с параметрами расчета, dict
        :param freq_q_ag: частота вращения ЭЦН или расход закачки газлифтного газа, Гц или ст. м3/с, float
        :param equipment_data: словарь с параметрами оборудования, dict
        :param well_trajectory_data: словарь с таблицей с инклинометрией, dict
        :param temperature_option: опция для расчета температуры, 'Const' или 'Linear', string
        :param calc_type: цель расчета (Что мы хотим посчитать и сравнить? 'well', 'pvt', 'esp', 'pipe'), str
        :param result_path: путь к файлу с результатами, str
        :param heat_balance: True - учет теплопотерь, False - расчет без учета теплопотерь
        :param h_start: 'top' или 'bottom'
        :param flow_direction: -1 - расчет от h_start,
                                1 - расчет к h_start
        :param mean_integral_error: True - среднеинтегральная ошибка,
                                    False - относительная ошибка
        :param ambient_temperature_data: распределение температуры по стволу скважины, dict
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
                                         mean_integral_error=mean_integral_error)
            except IndexError:
                continue

            if np.isnan(results_dict['error_results']['P_atma']):
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
        results_df = self.reformat_results(results_df, calc_type)
        if not calc_options['scenario']:
            if len(results_df) > 0:
                # results_df = self.reformat_results(results_df, calc_type)
                results_df.to_excel(result_path)

        return results_df

    def calc_model_uniflocpy(self,
                             fluid_data: dict,
                             pipe_data: dict,
                             pfl: float = None,
                             equipment_data: dict = None,
                             well_trajectory_data: dict = None,
                             profile_variables: list = None,
                             calculation_type: str = 'pipe',
                             heat_balance=None,
                             flow_direction=None,
                             h_start=None):

        if equipment_data is None:
            equipment_data = {}

        if profile_variables is None:
            profile_variables = self.profile_variables

        q_liq = fluid_data['q_fluid']
        wct = fluid_data['wct']

        if calculation_type.lower() == 'pipe':

            # Обновление дополнительных параметров для инициализации Pipeline
            pvt = FluidFlow(**fluid_data)
            well_trajectory = Trajectory(**well_trajectory_data)
            pipe_data['tubing'].update({'fluid': pvt, 'trajectory': well_trajectory, 'top_depth': 0})

            # Инициализация Pipeline
            if pipe_data['tubing']['bottom_depth'] < pipe_data['casing']['bottom_depth'] or \
                pipe_data['tubing']['bottom_depth'] > pipe_data['casing']['bottom_depth']:
                raise ValueError('Для расчета трубы(pipe), длина НКТ должна быть равна длине обсадной колонны')

            # if pipe_data['tubing']['d'] > pipe_data['casing']['d']:
            #     raise ValueError('Для расчета трубы(pipe), диаметр НКТ не должен превышать диаметр обсадной колонны')

            pipe_data_new = deepcopy(pipe_data)
            pipe = Pipeline(**pipe_data_new['tubing'])

            # Расчет трубы
            p_start = pfl
            t_start = self.TEMPERATURE_INDEX[0]
            # Проверим, что в массиве глубин нет противоречий
            if self.DEPTH[-1] > pipe.bottom_depth:
                self.DEPTH[-1] = pipe.bottom_depth

            if flow_direction == 1 and h_start == 'top':
                depth = sorted(self.DEPTH, reverse=False)
            else:
                raise ValueError('Поддерживается расчет только от линейного давления (h_start=top, flow_direction=1)')

            # steps = [i for i in range(1, pipe_data['tubing']['bottom_depth'] + 1)]
            pipe.calc_pt(q_liq=q_liq,
                         wct=wct,
                         h_start=h_start,
                         p_mes=p_start,
                         t_mes=t_start,
                         flow_direction=flow_direction,
                         step_len=self.L_REPORT,
                         hydr_corr_type=self.hydr_corr_type,
                         heat_balance=heat_balance,
                         extra_output=['dp_dl', 'dp_dl_grav', 'dp_dl_fric', 'dp_dl_acc', 'vsl',
                                       'liquid_holdup', 'flow_pattern', 'lambda_l', 'vsg',
                                       'n_re', 'angle'],
                         steps=depth)
            # angle_list = []
            # angle_step = 0
            # for i in range(1, pipe_data['tubing']['bottom_depth'] + 1):
            #     angle_list.append(well_trajectory.calc_angle(angle_step, angle_step+1))
            #     angle_step += 1
            #
            # return {'angle': pd.DataFrame(index=pipe.distributions['depth'],
            #                               data=np.array(angle_list))}
            return {
                'P_atma': pd.DataFrame(index=pipe.distributions['depth'],
                                       data=np.array(pipe.distributions['p']) / 101325),
                'Temperature': pd.DataFrame(index=pipe.distributions['depth'],
                                            data=np.array(pipe.distributions['t']) - 273.15),
                'Liquid_velocity': pd.DataFrame(index=pipe.distributions['depth'],
                                                data=pipe.distributions['vsl']),
                'dP_dL': pd.DataFrame(index=pipe.distributions['depth'],
                                      data=np.array(pipe.distributions['dp_dl']) / 101325),
                'dP_dL_fric': pd.DataFrame(index=pipe.distributions['depth'],
                                           data=np.array(pipe.distributions['dp_dl_fric']) / 101325),
                'dP_dL_grav': pd.DataFrame(index=pipe.distributions['depth'],
                                           data=np.array(pipe.distributions['dp_dl_grav']) / 101325),
                'dP_dL_acc': pd.DataFrame(index=pipe.distributions['depth'],
                                          data=np.array(pipe.distributions['dp_dl_acc']) / 101325),
                'Liquid_holdup': pd.DataFrame(index=pipe.distributions['depth'],
                                              data=pipe.distributions['liquid_holdup']),
                'Lambda': pd.DataFrame(index=pipe.distributions['depth'],
                                       data=pipe.distributions['lambda_l']),
                'n_re': pd.DataFrame(index=pipe.distributions['depth'],
                                     data=pipe.distributions['n_re']),
                # 'angle': pd.DataFrame(index=pipe.distributions['depth'],
                #                      data=pipe.distributions['angle'])
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

        if equipment_data is not None:
            equipment_data_pipesim = copy.deepcopy(equipment_data)

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
            self.calc_model_uniflocpy(pfl=uc.convert_pressure(pfl, 'atm', 'pa'),
                                      fluid_data=fluid_data_new,
                                      pipe_data=pipe_data,
                                      equipment_data=equipment_data,
                                      well_trajectory_data=well_trajectory_data,
                                      calculation_type=calc_type,
                                      heat_balance=heat_balance,
                                      flow_direction=flow_direction,
                                      h_start=h_start)

        # Приведем результаты Pipesim к единому с UniflocPy формату
        pipesim_results_f = self.formate_pipesim_results(pipesim_results, calc_type, list(uniflocpy_results.keys()))

        if calc_options['error_calc']:
            # Запустим расчет ошибки
            error_results = self.calc_error(uniflocpy_results, pipesim_results_f, mean_integral_error)
        else:
            error_results = None

        if calc_options['save_results']:
            self.save_results(uniflocpy_results, pipesim_results_f, error_results, result_path, calc_type=calc_type)

        if calc_options['plot_results']:
            self.plot_results(uniflocpy_results, pipesim_results_f, calc_type)

        return {'pipesim_results': pipesim_results_f, 'uniflocpy_results': uniflocpy_results,
                'error_results': error_results, 'density_inversion': flag}

