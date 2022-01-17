import pandas as pd
from typing import Tuple, Union, Any, Optional
from sixgill.definitions import Parameters, ProfileVariables
import numpy as np
from unifloc.tools import units_converter as uc
import matplotlib.pyplot as plt
import math as mt
from scipy.interpolate import interp1d
from copy import deepcopy


class Autotest:
    def __init__(self,
                 model_path,
                 file_path,
                 trajectory_data,
                 ambient_temperature_data,
                 fluid_data,
                 pipe_data,
                 equipment_data):

        self.model_path = model_path
        self.file_path = file_path
        self.trajectory_data = trajectory_data
        self.ambient_temperature_data = ambient_temperature_data
        self.fluid_data = fluid_data
        self.pipe_data = pipe_data
        self.equipment_data = equipment_data

        # Глобальные параметры
        self.PRESSURE_INDEX = None  # Массив давлений при сравнении PVT
        self.TEMPERATURE_INDEX = None  # Массив температур при сравнении PVT
        self.DEPTH = None
        self.Q_MIX = None
        self.P_IN = None
        self.ESP_OLD = True
        self.L_REPORT = 10  # Шаг вывода параметров КРД, м
        self.P_RES = 99000000 / (10 ** 5)  # Пластовое давление, бара
        self.T_WH = 20  # Температура на буфере скважины, С
        self.PI = 1  # Коэффициент продуктивности, м3/сут/бар

        self.profile_variables = [ProfileVariables.TEMPERATURE,
                                  ProfileVariables.PRESSURE,
                                  ProfileVariables.MEASURED_DEPTH,
                                  ProfileVariables.DENSITY_GAS_INSITU,
                                  ProfileVariables.DENSITY_OIL_INSITU,
                                  ProfileVariables.DENSITY_WATER_INSITU,
                                  ProfileVariables.LIVE_OIL_SATURATED_VISCOSITY_INSITU,
                                  ProfileVariables.OIL_FORMATION_VOLUME_FACTOR,
                                  ProfileVariables.COMPRESSIBILITY_OIL_INSITU,
                                  ProfileVariables.BUBBLE_POINT_PRESSURE_INSITU,
                                  ProfileVariables.SOLUTION_GAS_IN_OIL_INSITU,
                                  ProfileVariables.VISCOSITY_DEAD_OIL_STOCKTANK,
                                  ProfileVariables.VISCOSITY_GAS_INSITU,
                                  ProfileVariables.VISCOSITY_OIL_INSITU,
                                  ProfileVariables.VISCOSITY_WATER_INSITU,
                                  ProfileVariables.Z_FACTOR_GAS_INSITU,
                                  ProfileVariables.SUPERFICIAL_VELOCITY_LIQUID,
                                  ProfileVariables.PRESSURE_GRADIENT_TOTAL,
                                  ProfileVariables.PRESSURE_GRADIENT_FRICTION,
                                  ProfileVariables.PRESSURE_GRADIENT_ELEVATION,
                                  ProfileVariables.PRESSURE_GRADIENT_ACCELERATION,
                                  ProfileVariables.HOLDUP_FRACTION_LIQUID,
                                  ProfileVariables.FROUDE_NUMBER_LIQUID,
                                  ProfileVariables.FLOW_PATTERN_GAS_LIQUID,
                                  ProfileVariables.VOLUME_FRACTION_LIQUID,
                                  ProfileVariables.SURFACE_TENSION_OIL_GAS_INSITU,
                                  ProfileVariables.SURFACE_TENSION_LIQUID_INSITU,
                                  ProfileVariables.SURFACE_TENSION_WATER_GAS_INSITU,
                                  ProfileVariables.VOLUME_FLOWRATE_OIL_INSITU,
                                  ProfileVariables.VOLUME_FLOWRATE_GAS_INSITU,
                                  ProfileVariables.VOLUME_FLOWRATE_WATER_INSITU,
                                  ProfileVariables.VOLUME_FLOWRATE_FLUID_INSITU,
                                  ProfileVariables.VOLUME_FLOWRATE_LIQUID_INSITU,
                                  ProfileVariables.SUPERFICIAL_VELOCITY_GAS,
                                  ProfileVariables.DENSITY_LIQUID_INSITU,
                                  ProfileVariables.DENSITY_FLUID_NO_SLIP_INSITU,
                                  ProfileVariables.VISCOSITY_FLUID_NO_SLIP_INSITU,
                                  ProfileVariables.VISCOSITY_LIQUID_INSITU,
                                  ProfileVariables.REYNOLDS_NUMBER,
                                  ProfileVariables.PIPE_ANGLE_TO_HORIZONTAL,
                                  ProfileVariables.TEMPERATURE,
                                  ProfileVariables.TEMPERATURE_GRADIENT_OVERALL,
                                  ProfileVariables.TEMPERATURE_GRADIENT_JOULE_THOMSON,
                                  ProfileVariables.TEMPERATURE_GRADIENT_HEAT_TRANSFER,
                                  ProfileVariables.TEMPERATURE_GRADIENT_ELEVATION,
                                  ProfileVariables.JOULE_THOMPSON_COEFFICIENT_INSITU,
                                  ProfileVariables.OVERALL_HEAT_TRANSFER_COEFFICIENT,
                                  ProfileVariables.INSIDE_FILM_FORCED_CONV_NUSSELT_NUMBER,
                                  ProfileVariables.INSIDE_FILM_NATURAL_CONV_NUSSELT_NUMBER,
                                  ProfileVariables.HEAT_CAPACITY_GAS_INSITU,
                                  ProfileVariables.HEAT_CAPACITY_LIQUID_INSITU,
                                  ProfileVariables.INSIDE_FILM_NUSSELT_NUMBER,
                                  ProfileVariables.INSIDE_FILM_GAS_REYNOLDS_NUMBER,
                                  ProfileVariables.INSIDE_FILM_LIQUID_REYNOLDS_NUMBER,
                                  ProfileVariables.HEAT_CAPACITY_FLUID_INSITU,
                                  ProfileVariables.HEAT_CAPACITY_GAS_INSITU,
                                  ProfileVariables.HEAT_CAPACITY_WATER_INSITU,
                                  ProfileVariables.HEAT_CAPACITY_OIL_INSITU,
                                  ProfileVariables.CASING_GAS_PRESSURE]

        self.results_mapping = {'bo': ProfileVariables.OIL_FORMATION_VOLUME_FACTOR,
                                'pb': ProfileVariables.BUBBLE_POINT_PRESSURE_INSITU,
                                'ro': ProfileVariables.DENSITY_OIL_INSITU,
                                'muo': ProfileVariables.VISCOSITY_OIL_INSITU,
                                'mug': ProfileVariables.VISCOSITY_GAS_INSITU,
                                'muw': ProfileVariables.VISCOSITY_WATER_INSITU,
                                'rw': ProfileVariables.DENSITY_WATER_INSITU,
                                'rg': ProfileVariables.DENSITY_GAS_INSITU,
                                'z': ProfileVariables.Z_FACTOR_GAS_INSITU,
                                'rs': ProfileVariables.SOLUTION_GAS_IN_OIL_INSITU,
                                'co': ProfileVariables.COMPRESSIBILITY_OIL_INSITU,
                                'P_atma': ProfileVariables.PRESSURE,
                                't': ProfileVariables.TEMPERATURE,
                                'p_cas': ProfileVariables.CASING_GAS_PRESSURE,
                                'depth': ProfileVariables.MEASURED_DEPTH,
                                'Liquid_velocity': ProfileVariables.SUPERFICIAL_VELOCITY_LIQUID,
                                'dP_dL': ProfileVariables.PRESSURE_GRADIENT_TOTAL,
                                'dP_dL_fric': ProfileVariables.PRESSURE_GRADIENT_FRICTION,
                                'dP_dL_grav': ProfileVariables.PRESSURE_GRADIENT_ELEVATION,
                                'dP_dL_acc': ProfileVariables.PRESSURE_GRADIENT_ACCELERATION,
                                'Temperature': ProfileVariables.TEMPERATURE,
                                'Liquid_holdup': ProfileVariables.HOLDUP_FRACTION_LIQUID,
                                'Froude': ProfileVariables.FROUDE_NUMBER_LIQUID,
                                'Flow_pattern': ProfileVariables.FLOW_PATTERN_GAS_LIQUID,
                                'Lambda': ProfileVariables.VOLUME_FRACTION_LIQUID,
                                'esp_dp': 'ESP delta P',
                                'esp_head': 'ESP Head',
                                'esp_power': 'Power',
                                'esp_eff': 'ESP Efficiency',
                                'stog': ProfileVariables.SURFACE_TENSION_OIL_GAS_INSITU,
                                'stwg': ProfileVariables.SURFACE_TENSION_WATER_GAS_INSITU,
                                'stlg': ProfileVariables.SURFACE_TENSION_LIQUID_INSITU,
                                'esp_p_dis': 'Pressure',
                                'qo': ProfileVariables.VOLUME_FLOWRATE_OIL_INSITU,
                                'qg': ProfileVariables.VOLUME_FLOWRATE_GAS_INSITU,
                                'qw': ProfileVariables.VOLUME_FLOWRATE_WATER_INSITU,
                                'ql': ProfileVariables.VOLUME_FLOWRATE_LIQUID_INSITU,
                                'qm': ProfileVariables.VOLUME_FLOWRATE_FLUID_INSITU,
                                'Gas_velocity': ProfileVariables.SUPERFICIAL_VELOCITY_GAS,
                                'rl': ProfileVariables.DENSITY_LIQUID_INSITU,
                                'rm': ProfileVariables.DENSITY_FLUID_NO_SLIP_INSITU,
                                'mum': ProfileVariables.VISCOSITY_FLUID_NO_SLIP_INSITU,
                                'mul': ProfileVariables.VISCOSITY_LIQUID_INSITU,
                                'n_re': ProfileVariables.REYNOLDS_NUMBER,
                                'angle': ProfileVariables.PIPE_ANGLE_TO_HORIZONTAL,
                                'Nusselt': ProfileVariables.INSIDE_FILM_NUSSELT_NUMBER,
                                'Nusselt Free': ProfileVariables.INSIDE_FILM_NATURAL_CONV_NUSSELT_NUMBER,
                                'Nusslet Forced': ProfileVariables.INSIDE_FILM_FORCED_CONV_NUSSELT_NUMBER,
                                're_l': ProfileVariables.INSIDE_FILM_LIQUID_REYNOLDS_NUMBER,
                                're_g': ProfileVariables.INSIDE_FILM_GAS_REYNOLDS_NUMBER,
                                'jt': ProfileVariables.JOULE_THOMPSON_COEFFICIENT_INSITU,
                                'T_G_JT': ProfileVariables.TEMPERATURE_GRADIENT_JOULE_THOMSON,
                                'T_G_HT': ProfileVariables.TEMPERATURE_GRADIENT_HEAT_TRANSFER,
                                'T_G_Elev': ProfileVariables.TEMPERATURE_GRADIENT_ELEVATION,
                                'cp_n': ProfileVariables.HEAT_CAPACITY_FLUID_INSITU,
                                'cp_l': ProfileVariables.HEAT_CAPACITY_LIQUID_INSITU,
                                'cp_g': ProfileVariables.HEAT_CAPACITY_GAS_INSITU,
                                'u': ProfileVariables.OVERALL_HEAT_TRANSFER_COEFFICIENT,
                                'dt_dl_ground_coef': ProfileVariables.TEMPERATURE_GRADIENT_GROUND_AND_AMBIENT,
                                'dt_dl': ProfileVariables.TEMPERATURE_GRADIENT_OVERALL
                                }

    def calc_error_one_array(self, uniflocpy_array, pipesim_array, mean_integral_error, key):
        """

        Parameters
        ----------
        :param uniflocpy_array: массив uniflocpy
        :param pipesim_array: массив pipesim

        Returns
        -------

        """
        global DEPTH
        if isinstance(uniflocpy_array, pd.DataFrame) and isinstance(pipesim_array, pd.DataFrame):

            # Среднеинтегральная ошибка
            if mean_integral_error:
                relative_error = []
                for i in range(len(pipesim_array.values)):
                    relative = abs(pipesim_array.values[i] - uniflocpy_array[::-1].values[i]) / abs(
                        pipesim_array.values[i])
                    relative_error.append(relative[0])
                error = np.trapz(relative_error, x=DEPTH) / abs(DEPTH[0] - DEPTH[-1])
            # Относительная ошибка
            else:
                error = abs(uniflocpy_array[uniflocpy_array.index.notnull()].mean().values[0] -
                            pipesim_array[pipesim_array.index.notnull()].mean().values[0]) / \
                        abs(pipesim_array[pipesim_array.index.notnull()].mean().values[0])
        elif isinstance(uniflocpy_array, list) and isinstance(pipesim_array, list):
            uniflocpy_array = np.array(uniflocpy_array)
            pipesim_array = np.array(pipesim_array)
            error = abs(uniflocpy_array.mean() - pipesim_array.mean()) / abs(pipesim_array.mean())
        else:
            error = abs(uniflocpy_array - pipesim_array) / abs(pipesim_array)
        return error * 100

    def calc_error(self, uniflocpy_results=None, pipesim_results=None, mean_integral_error=None) -> dict:
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
                error = self.calc_error_one_array(uniflocpy_results[key], pipesim_results[key], mean_integral_error,
                                                  key)

                # Сохраним в словарь средних ошибок
                error_dict.update({key: [error]})
        else:
            print('Один из массивов не заполнен, невозможно рассчитать ошибку')
        return error_dict

    def find_key_in_dict(self, key: str, data: float, dic: dict, limited_pars: dict) -> Tuple[dict, bool]:
        """
        Функция для поиска ключа в словаре

        Parameters
        ----------
        :param key: ключ для поиска в словаре, string
        :param data: значение параметра, float
        :param dic: словарь для поиска, dict
        :param limited_pars: словарь параметров с ограничениями, dict
        Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing

        Returns
        -------

        """
        dic_new = dic.copy()
        flag_find = False

        if dic is not None:
            if isinstance(dic, dict):
                # В первую очередь проверим, есть ли ключ в словаре ограничений

                # Проверим накладывается ли на выбранный ключ ограничение на наличие в определенном словаре
                if limited_pars is not None:
                    if key in limited_pars.keys():
                        # Проверим есть ли в текущем словаре наш определенный словарь
                        if limited_pars[key] in dic.keys():
                            flag_find = True
                            dic_new[limited_pars[key]][key] = data
                            return dic_new, flag_find

                if key in dic.keys():
                    dic_new[key] = data
                    flag_find = True
                else:
                    # Оставим словарь, в котором только словари
                    for key_child, dic_child in {k: v for k, v in dic.items() if isinstance(v, dict)}.items():

                        # Рекурсивно будем искать словарь в словаре
                        dic_child_new, flag_find = self.find_key_in_dict(key, data, dic_child, limited_pars)

                        # Если найден, то вернем в новый словарь
                        if flag_find:
                            dic_new[key_child] = dic_child_new

        return dic_new, flag_find

    def find_change_parameter(self, keys: list, data: list, pfl: float, model_path: str, fluid_data: dict,
                              pipe_data: dict, equipment_data: dict, limited_pars: dict) -> \
            Tuple[Union[float, Any], Union[float, Any], str, dict, dict, Optional[dict]]:
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
        freq_q_ag_new = None
        fluid_data_new = None
        pipe_data_new = None
        equipment_data_new = None
        init_dicts = [fluid_data, pipe_data, equipment_data]

        # Цикл поиска ключей в исходных данных
        for i in range(len(keys)):
            if keys[i] == 'pfl':
                pfl_new = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 'freq_q_ag':
                freq_q_ag_new = data[i]
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

        if pfl_new is None:
            pfl_new = pfl

        if fluid_data_new is None:
            fluid_data_new = fluid_data

        if pipe_data_new is None:
            pipe_data_new = pipe_data

        if equipment_data_new is None:
            equipment_data_new = equipment_data

        return pfl_new, freq_q_ag_new, model_path_new, fluid_data_new, pipe_data_new, equipment_data_new

    def analyze_density_inversion(self, results):
        """
        Функция для проверки на превышение плотности газа над нефтью

        :param results: результаты
        :return: флаг - была ошибка или нет
        """
        if len(results.summary['Warning']) > 0:
            for warning in results.summary['Warning']:
                warning_split = warning[warning.find('Warning') + 9:]
                warning_number = warning_split[:warning_split.find(' ')]
                if warning_number == '7004072':
                    return 1
                else:
                    continue
            else:
                return 0
        else:
            return 0

    def replace_num_flow_pattern(self, flow_pattern):
        """
        Функция для замены номера режима на соответствующий номер режима из UniflocPy

        Parameters
        ----------
        :param flow_pattern: номер режима в Pipesim

        Returns
        -------
        """
        if flow_pattern == 5:
            flow_pattern = 0
        elif flow_pattern == 7:
            flow_pattern = 1
        elif flow_pattern == 8:
            flow_pattern = 2
        elif flow_pattern == 6:
            flow_pattern = 3
        return flow_pattern

    def define_pvt_table_data(self, pipesim_results, pars, table_model_data):
        """
        Функция для подготовки табличной PVT

        Parameters
        ----------
        :param pipesim_results: результаты  для извлечения
        :param pars: параметры для вывода
        :param table_model_data: исходный словарь с табличной PVT

        Returns
        -------
        table_model_data : обновленный словарь с табличной PVT
        """
        pressure_array = uc.convert_pressure(pipesim_results.profile[pipesim_results.cases[0]]
                                             ['Pressure'], 'bar', 'pa')[3:]

        for i in range(len(pars)):
            if pars[i] == 'pb':
                data = list(uc.convert_pressure(pipesim_results.profile[pipesim_results.cases[0]]
                                                [self.results_mapping[pars[i]]], 'bar', 'pa'))[3:]
            else:
                data = pipesim_results.profile[pipesim_results.cases[0]][self.results_mapping[pars[i]]][3:]

            df = pd.DataFrame(index=pressure_array, data=data, columns=[pars[i]])
            table_model_data.update({pars[i]: df})
        return table_model_data

    def formate_pipesim_results(self, pipesim_results=None, calc_type: str = 'well', pars=None) -> dict:
        """
        Функция для форматирования результатов из Pipesim

        Parameters
        ----------
        :param pars: параметры для вывода из Pipesim
        :param pipesim_results: результаты из Pipesim
        :param calc_type: тип расчета

        Returns
        -------

        """
        if pars is None:
            pars = ['p_distribution']

        pipesim_results_dict = {}

        if pipesim_results is not None:
            if calc_type.lower() == 'well':
                for i in range(len(pars)):
                    # Создадим таблицу профиля параметра по глубине
                    if pars[i] == 'P_atma':
                        data = uc.convert_pressure(pipesim_results.profile[pipesim_results.cases[0]]
                                                   [self.results_mapping[pars[i]]], 'bar', 'atm')
                    else:
                        data = pipesim_results.profile[pipesim_results.cases[0]][self.results_mapping[pars[i]]]
                    par_df = pd.DataFrame(index=pipesim_results.profile[pipesim_results.cases[0]][ProfileVariables.
                                          MEASURED_DEPTH][:1:-1], data=data[:1:-1], columns=[pars[i]])

                    pipesim_results_dict.update({pars[i]: par_df})

            elif calc_type.lower() == 'pvt':
                for i in range(len(pars)):
                    if pars[i] == 'pb':
                        data = list(uc.convert_pressure(pipesim_results.profile[pipesim_results.cases[0]]
                                                        [self.results_mapping[pars[i]]], 'bar', 'mpa'))[3:]
                    elif pars[i] == 'stog' or pars[i] == 'stwg' or pars[i] == 'stlg':
                        data = [par / 1000 for par in
                                pipesim_results.profile[pipesim_results.cases[0]][self.results_mapping[pars[i]]][3:]]
                    else:
                        data = pipesim_results.profile[pipesim_results.cases[0]][self.results_mapping[pars[i]]][3:]

                    pipesim_results_dict.update({pars[i]: data})
            elif calc_type.lower() == 'pipe':
                for i in range(len(pars)):
                    if pars[i] == 'P_atma' or pars[i] == 'dP_dL' or pars[i] == 'dP_dL_grav' or pars[
                        i] == 'dP_dL_fric' or \
                            pars[i] == 'dP_dL_acc':
                        data = uc.convert_pressure(pipesim_results.profile[pipesim_results.cases[0]]
                                                   [self.results_mapping[pars[i]]], 'bar', 'atm')
                    elif pars[i] == 'Flow_pattern':
                        data = pipesim_results.profile[pipesim_results.cases[0]][self.results_mapping[pars[i]]]
                        data = [self.replace_num_flow_pattern(pattr) for pattr in data]
                    else:
                        data = pipesim_results.profile[pipesim_results.cases[0]][self.results_mapping[pars[i]]]

                    par_df = pd.DataFrame(index=pipesim_results.profile[pipesim_results.cases[0]][ProfileVariables.
                                          MEASURED_DEPTH][:1:-1], data=data[:1:-1],
                                          columns=[pars[i]])

                    pipesim_results_dict.update({pars[i]: par_df})
            else:
                for i in range(len(pars)):

                    # Определим переменную с переводным коэффициентом
                    if pars[i] == 'esp_dp':
                        pipesim_par = uc.convert_pressure(pipesim_results.system[self.results_mapping[pars[i]]][
                                                              pipesim_results.cases[0]], 'bar', 'atm')
                    elif pars[i] == 'q_mix':
                        pipesim_par = self.Q_MIX
                    elif pars[i] == 'esp_power':
                        multiplier = 1000
                        pipesim_par = pipesim_results.node[pipesim_results.cases[0]][self.results_mapping[pars[i]]][
                            'Esp 1']
                        pipesim_par *= multiplier
                    elif pars[i] == 'esp_p_dis':
                        pipesim_par = uc.convert_pressure(pipesim_results.node[pipesim_results.cases[0]]
                                                          [self.results_mapping[pars[i]]]['Esp 1'], 'bar', 'atm')
                    else:
                        pipesim_par = pipesim_results.system[self.results_mapping[pars[i]]][pipesim_results.cases[0]]

                    pipesim_results_dict.update({pars[i]: [pipesim_par]})
        else:
            print('Результаты Pipesim пустые')

        return pipesim_results_dict

    def reformat_results(self, df, calc_type):
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

            if calc_type != 'esp':
                # ! Не понимаю почему в esp не добавляются в ячейку как список
                error_dict = error_dict[0]

            for k, v in error_dict.items():
                df.loc[i, 'Error ' + k] = v

        del df['Error']
        return df

    def save_results(self, uniflocpy_results=None, pipesim_results=None, error_results=None,
                     file_path="C:/Users/PC/PycharmProjects/pythonProject/hagerdon_results/test new autotest.xlsx",
                     calc_type: str = 'pipe'):
        """
        Функция для сохранения результатов в Excel

        Parameters
        ----------
        :param calc_type : тип расчета
        :param uniflocpy_results : результаты из UniflocPy
        :param pipesim_results : результаты из Pipesim
        :param error_results: словарь с средними ошибками
        :param file_path : путь к файлу с результатами

        Returns
        -------

        """

        if uniflocpy_results is None and pipesim_results is None and error_results is None:
            print('Массивы результатов и ошибки пустые')
            return

        if error_results is not None:
            error_df = pd.DataFrame.from_dict(error_results)
        else:
            error_df = None

        if uniflocpy_results is not None:
            if calc_type.lower() == 'pvt':
                uniflocpy_results_df = pd.DataFrame(uniflocpy_results, index=self.PRESSURE_INDEX)
            elif calc_type.lower() == 'pipe' or calc_type.lower() == 'well':
                uniflocpy_results_df = pd.concat(uniflocpy_results.values(), axis=1)
            else:
                uniflocpy_results_df = pd.DataFrame.from_dict(uniflocpy_results)
        else:
            uniflocpy_results_df = None

        if pipesim_results is not None:
            if calc_type.lower() == 'pvt':
                pipesim_results_df = pd.DataFrame(pipesim_results, index=self.PRESSURE_INDEX)
            elif calc_type.lower() == 'pipe' or calc_type.lower() == 'well':
                pipesim_results_df = pd.concat(pipesim_results.values(), axis=1)
            else:
                pipesim_results_df = pd.DataFrame.from_dict(pipesim_results)
        else:
            pipesim_results_df = None

        # Сохраним результаты в 3 листа
        with pd.ExcelWriter(file_path) as w:
            if error_df is not None:
                error_df.to_excel(w, sheet_name='Error')
            if uniflocpy_results_df is not None:
                uniflocpy_results_df.to_excel(w, sheet_name='UniflocPy')
            if pipesim_results_df is not None:
                pipesim_results_df.to_excel(w, sheet_name='Pipesim')
        return

    def make_subplots(self, xs_pipesim, ys_pipesim, xs_uniflocpy, ys_uniflocpy):
        """
        Функция для построения графиков

        Parameters
        ----------
        xs_pipesim : оси X для графиков из Pipesim
        ys_pipesim : оси Y - значения свойств зависящих от X в Pipesim
        xs_uniflocpy : оси X для графиков из UniflocPy
        ys_uniflocpy : оси Y - значения свойств зависящих от X в UniflocPy

        Returns
        -------

        """
        plt.rc('font', size=5)  # controls default text sizes
        if isinstance(xs_pipesim, dict):
            number_of_graphs = len(xs_pipesim)
            nx = mt.ceil(number_of_graphs / 5)
            ny = mt.ceil(number_of_graphs / nx)

            if nx != 1:
                iterate_array = np.nditer(np.zeros((nx, ny)), flags=['multi_index'])
            else:
                iterate_array = np.arange(ny)

            fig, axes = plt.subplots(nx, ny, sharey=True)
            flag_invertion = True
            for key, j in zip(xs_pipesim.keys(), iterate_array):
                if isinstance(xs_pipesim[key], pd.DataFrame):
                    x_pipesim = xs_pipesim[key].iloc[:, 0].values
                    x_uniflocpy = xs_uniflocpy[key].iloc[:, 0].values
                else:
                    x_pipesim = xs_pipesim[key]
                    x_uniflocpy = xs_uniflocpy[key]

                if len(iterate_array.shape) > 1:
                    idx = iterate_array.multi_index
                else:
                    idx = j

                if number_of_graphs != 1:
                    axes[idx].plot(x_pipesim, ys_pipesim, 'b-', label='pipesim')
                    axes[idx].plot(x_uniflocpy, ys_uniflocpy, 'c--', label='uniflocpy')
                    axes[idx].set_title(key)
                    axes[idx].set_ylabel('MD, м')
                    axes[idx].legend()
                    if flag_invertion:
                        axes[idx].invert_yaxis()
                        flag_invertion = False
                    axes[idx].grid()
                else:
                    axes.plot(x_pipesim, ys_pipesim, 'b-', label='pipesim')
                    axes.plot(x_uniflocpy, ys_uniflocpy, 'c--', label='uniflocpy')
                    axes.set_title(key)
                    axes.set_ylabel('MD, м')
                    if flag_invertion:
                        axes.invert_yaxis()
                        flag_invertion = False
                    axes.legend()
                    axes.grid()

        else:
            number_of_graphs = len(ys_pipesim)
            nx = mt.ceil(number_of_graphs / 5)
            ny = mt.ceil(number_of_graphs / nx)

            if nx != 1:
                iterate_array = np.nditer(np.zeros((nx, ny)), flags=['multi_index'])
            else:
                iterate_array = np.arange(ny)

            fig, axes = plt.subplots(nx, ny, sharex=True)

            for key, j in zip(ys_pipesim.keys(), iterate_array):
                if isinstance(ys_pipesim[key], pd.DataFrame):
                    y_pipesim = ys_pipesim[key].iloc[:, 0].values
                    y_uniflocpy = ys_uniflocpy[key].iloc[:, 0].values
                else:
                    y_pipesim = ys_pipesim[key]
                    y_uniflocpy = ys_uniflocpy[key]

                if len(iterate_array.shape) > 1:
                    idx = iterate_array.multi_index
                else:
                    idx = j

                if number_of_graphs != 1:
                    axes[idx].plot(xs_pipesim, y_pipesim, 'b-', label='pipesim')
                    axes[idx].plot(xs_uniflocpy, y_uniflocpy, 'c--', label='uniflocpy')
                    axes[idx].set_xlabel('P, атма')
                    axes[idx].set_ylabel(key)
                    axes[idx].legend()
                    axes[idx].grid()
                else:
                    axes.plot(xs_pipesim, y_pipesim, 'b-', label='pipesim')
                    axes.plot(xs_uniflocpy, y_uniflocpy, 'c--', label='uniflocpy')
                    axes.set_xlabel('P, атма')
                    axes.set_ylabel(key)
                    axes.legend()
                    axes.grid()

        plt.show()

    def plot_results(self, uniflocpy_results, pipesim_results, calc_type):
        """
        Функция для визуализации результатов

        Parameters
        ----------
        :param uniflocpy_results : результаты из UniflocPy
        :param pipesim_results : результаты из Pipesim
        :param calc_type : тип расчета

        Returns
        -------
        """

        if calc_type.lower() == 'pvt':
            self.make_subplots(xs_pipesim=self.PRESSURE_INDEX, ys_pipesim=pipesim_results,
                               xs_uniflocpy=self.PRESSURE_INDEX, ys_uniflocpy=uniflocpy_results)
        elif calc_type.lower() == 'esp':
            print('Значения дискретные. График нет смысла строить')
        else:
            self.make_subplots(xs_pipesim=pipesim_results,
                               ys_pipesim=pipesim_results[list(pipesim_results.keys())[0]].index,
                               xs_uniflocpy=uniflocpy_results,
                               ys_uniflocpy=uniflocpy_results[list(uniflocpy_results.keys())[0]].index)

    def use_table_model(self,
                        black_oil_model,
                        h_cas,
                        parameters,
                        model,
                        ambient_temperature_data,
                        heat_balance_pipesim,
                        well_name,
                        h_tub,
                        pfl,
                        t_res,
                        fluid_data_new):

        if 'use_table_model' in black_oil_model:
            if black_oil_model['use_table_model']:
                props_to_generate = [key for key in black_oil_model['table_model_data'].keys()
                                     if black_oil_model['table_model_data'][key] is None]
                if len(props_to_generate) > 0:
                    # Увеличим диапазон глубин, чтобы увеличить диапазон давлений
                    updated_params = {'Tubing 1': {Parameters.Tubing.LENGTH: h_cas + 2000},
                                      'Casing 1': {Parameters.Casing.LENGTH: h_cas + 2000},
                                      'Vert Comp 1': {Parameters.Completion.TOPMEASUREDDEPTH: h_cas + 2000}}
                    parameters[Parameters.PTProfileSimulation.OUTLETPRESSURE] = 1
                    model.set_values(updated_params)
                    model.save()

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
                    t_array = np.arange(min(ambient_temperature_data['T']), max(ambient_temperature_data['T']) + 5, 5)

                    for t_res_local in t_array:
                        model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
                        geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, h_cas],
                                             Parameters.GeothermalSurvey.TEMPERATURE: [float(t_res_local - 273.15),
                                                                                       float(t_res_local - 273.15)]}
                        geothermal_df = pd.DataFrame(geothermal_survey)
                        model.set_geothermal_profile(Well=well_name, value=geothermal_df)

                        results = model.tasks.ptprofilesimulation.run(producer=well_name,
                                                                      parameters=parameters,
                                                                      profile_variables=self.profile_variables)
                        # вытаскиваем результаты
                        table_p_1d = self.define_pvt_table_data(results, props_to_generate,
                                                                black_oil_model['table_model_data'])

                        # найдем min и max значение давления на всех рассчетах
                        p_max = min(table_p_1d['bo'].index.values[0], p_max)
                        p_min = max(table_p_1d['bo'].index.values[-1], p_min)
                        p_array = np.arange(p_min, p_max, 10000)

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

                    # Вернем как было
                    updated_params = {'Tubing 1': {Parameters.Tubing.LENGTH: h_tub},
                                      'Casing 1': {Parameters.Casing.LENGTH: h_cas},
                                      'Vert Comp 1': {Parameters.Completion.TOPMEASUREDDEPTH: h_cas}}
                    parameters[Parameters.PTProfileSimulation.OUTLETPRESSURE] = uc.convert_pressure(pfl, 'pa', 'bar')
                    model.set_values(updated_params)

                    model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
                    geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, h_cas],
                                         Parameters.GeothermalSurvey.TEMPERATURE: [
                                             float(ambient_temperature_data['T'][0]) - 273.15,
                                             float(t_res)]}
                    geothermal_df = pd.DataFrame(geothermal_survey)
                    model.set_geothermal_profile(Well=well_name, value=geothermal_df)

    def pipesim_parameters(self,
                           wct,
                           model,
                           black_oil_model,
                           heat_balance,
                           temperature_option,
                           well_name,
                           t_res,
                           h_cas,
                           ambient_temperature_data,
                           constants,
                           equipment_data,
                           hydr_corr_type,
                           p,
                           qliq,
                           h_tub,
                           modelcomponents):
        # Случай чистой воды
        if wct == 100:
            updated_params = {'Black Oil 1': {Parameters.BlackOilFluid.USEGASRATIO: 'GLR',
                                              Parameters.BlackOilFluid.GLR: 0}}
            model.set_values(updated_params)

        # Проверка есть ли калибровочные значения параметров pb, rsb, bob, muob, задание если есть
        if 'rsb' in black_oil_model:
            if black_oil_model['rsb'] is not None:
                updated_params = {'Black Oil 1': {Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BUBBLEPOINTSATGAS_VALUE: black_oil_model['rsb']['value'],
                                                  Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BUBBLEPOINTSATGAS_PRESSURE:
                                                      uc.convert_pressure(black_oil_model['rsb']['p'], 'pa', 'bar'),
                                                  Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BUBBLEPOINTSATGAS_TEMPERATURE:
                                                      uc.convert_temperature(black_oil_model['rsb']['t'], 'k', 'c')}}
                model.set_values(updated_params)

        if 'bob' in black_oil_model:
            if black_oil_model['bob'] is not None:
                updated_params = {'Black Oil 1': {Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BELOWBBPOFVF_VALUE: black_oil_model['bob']['value'],
                                                  Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BELOWBBPOFVF_PRESSURE:
                                                      uc.convert_pressure(black_oil_model['bob']['p'], 'pa', 'bar'),
                                                  Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPOFVF_TEMPERATURE:
                                                      uc.convert_temperature(black_oil_model['bob']['t'], 'k', 'c')}}
                model.set_values(updated_params)

        if 'muob' in black_oil_model:
            if black_oil_model['muob'] is not None:
                updated_params = {'Black Oil 1': {Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BELOWBBPLIVEOILVISCOSITY_VALUE: black_oil_model['muob']['value'],
                                                  Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BELOWBBPLIVEOILVISCOSITY_TEMPERATURE:
                                                      uc.convert_temperature(black_oil_model['muob']['t'], 'k', 'c'),
                                                  Parameters.BlackOilFluid.SinglePointCalibration.
                                                      BELOWBBPLIVEOILVISCOSITY_PRESSURE:
                                                      uc.convert_pressure(black_oil_model['muob']['p'], 'pa', 'bar')}}
                model.set_values(updated_params)

        # Установим температуру
        if heat_balance:
            heat_balance_pipesim = 'HEAT BALANCE = ON'
        else:
            heat_balance_pipesim = 'HEAT BALANCE = OFF'
        if temperature_option == 'CONST':
            model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
            geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, h_cas],
                                 Parameters.GeothermalSurvey.TEMPERATURE: [float(t_res), float(t_res)]}
            geothermal_df = pd.DataFrame(geothermal_survey)
            model.set_geothermal_profile(Well=well_name, value=geothermal_df)

        elif temperature_option == 'LINEAR':
            model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
            geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, h_cas],
                                 Parameters.GeothermalSurvey.TEMPERATURE: [
                                     float(ambient_temperature_data['T'][0]) - 273.15,
                                     float(t_res)]}

            geothermal_df = pd.DataFrame(geothermal_survey)
            model.set_geothermal_profile(Well=well_name, value=geothermal_df)

        # Добавим заканчивание в скважину
        model.add(modelcomponents.COMPLETION, "Vert Comp 1", context=well_name,
                  parameters={Parameters.Completion.TOPMEASUREDDEPTH: h_cas,
                              Parameters.Completion.FLUIDENTRYTYPE: constants.CompletionFluidEntry.SINGLEPOINT,
                              Parameters.Completion.GEOMETRYPROFILETYPE: constants.Orientation.VERTICAL,
                              Parameters.Completion.IPRMODEL: constants.IPRModels.IPRPIMODEL,
                              Parameters.Completion.RESERVOIRPRESSURE: self.P_RES,
                              Parameters.IPRPIModel.LIQUIDPI: self.PI,
                              Parameters.Completion.RESERVOIRTEMPERATURE: t_res,
                              Parameters.Well.ASSOCIATEDBLACKOILFLUID: "Black Oil 1"})
        # Добавим пакер
        if 'packer' in equipment_data:
            if equipment_data['packer']:
                model.add(modelcomponents.PACKER, 'Packer 1', context=well_name,
                          parameters={Parameters.Packer.TOPMEASUREDDEPTH: h_tub - 1})

        # Установка гидравлической корреляции
        if hydr_corr_type.lower() == 'beggsbrill':
            model.sim_settings.global_flow_correlation(
                {Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})
        elif hydr_corr_type == 'Gray':
            model.sim_settings.global_flow_correlation(
                {Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.GRAY_MODIFIED,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})
        elif hydr_corr_type.lower() == 'hagedornbrown':
            model.sim_settings.global_flow_correlation(
                {Parameters.FlowCorrelation.SWAPANGLE: 0,
                 Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.TULSA,
                 Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                     constants.MultiphaseFlowCorrelation.TulsaLegacy.HAGEDORNBROWN_ORIGINAL,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})

        return [{Parameters.PTProfileSimulation.OUTLETPRESSURE: uc.convert_pressure(p, 'pa', 'bar'),
                Parameters.PTProfileSimulation.LIQUIDFLOWRATE: qliq,
                Parameters.PTProfileSimulation.FLOWRATETYPE: constants.FlowRateType.LIQUIDFLOWRATE,
                Parameters.PTProfileSimulation.CALCULATEDVARIABLE: constants.CalculatedVariable.INLETPRESSURE},
                heat_balance_pipesim]
