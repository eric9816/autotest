import pandas as pd
import xlwings as xw
from NEW_autotest.autotest_pipe_subclass import AutotestPipe
from NEW_autotest.autotest_pvt_subclass import AutotestPVT
from NEW_autotest.autotest_gaslift_subclass import AutotestGasLift
from NEW_autotest.autotest_esp_subclass import AutotestESP
from NEW_autotest.autotest_choke_subclass import AutotestChoke


def calc_autotest(p_atma: float,
                  file_path: str,
                  model_path: str,
                  trajectory_data: dict,
                  ambient_temperature_data: dict,
                  fluid_data: dict,
                  data: dict,
                  calc_type: str,
                  equipment_data: dict = None,
                  t_k: float = None,
                  esp_data: pd.DataFrame = None,
                  freq: float = None,
                  stage_by_stage: bool = False,
                  p_ann: float = None,
                  sample: bool = False,
                  number_of_samples: int = 5,
                  pars_limits: dict = None,
                  limited_pars: dict = None,
                  calc_options: dict = None,
                  hydr_corr_type: str = 'BeggsBrill',
                  q_gas_inj: float = None,
                  p_gas_inj: float = None,
                  esp_id: int = None,
                  heat_balance: bool = False,
                  temperature_option: str = 'LINEAR',
                  mean_integral_error: bool = False,
                  flow_direction: int = 1
                  ):
    """
    :param esp_data: паспортные данные ЭЦН - pd.Series или dict
    :param freq: частота ЭЦН
    :param stage_by_stage: расчет ЭЦН от ступени к ступени
    :param p_ann: давление в затрубном пространстве
    :param esp_id: id насоса
    :param hydr_corr_type: тип гидравлической корреляции
    :param p_atma: давление, атма
    :param file_path: путь сохранения файла с результатами
    :param model_path: путь сохранения файлов в формате pips
    :param trajectory_data: словарь с таблицей с инклинометрией, dict
    :param ambient_temperature_data: словарь с таблицей распределения температуры
    :param fluid_data: словарь с параметрами флюида, dict
    :param pipe_data: словарь с параметрами труб, dict
    :param equipment_data: словарь с параметрами оборудования, dict
    :param calc_type: тип расчета (pvt, well, pipe, esp)
    :param sample: если True, то расчет произведется на произвольном наборе данных.
    В противном случае рассчитается одиночный - main
    :param number_of_samples: количество набота данных (по умолчанию - 5)
    :param pars_limits: словарь с параметром и его границами, в которых будет генерироваться данные, dict
    :param limited_pars: словарь с однозначными соответствиями параметров для объектов, которые могут повторяться, dict
        Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing
    :param calc_options: словарь с параметрами расчета, dict
    :param q_gas_inj: закачка газлифтного газа, ст. м3/с (задается в случае если тип расчета well)
    :param p_gas_inj: давление закачки газлифтного газа, Па (задается в случае если тип расчета well)
    :param heat_balance:  опция учета теплопотерь, по умолчанию - False
    :param temperature_option: 'CONST' - темпертарура константа по всей скважине,
                               'LINEAR' - линейное распределение темпратуры
                               (по умолчанию - CONST)
    :param flow_direction: -1 - расчет от h_start,
                            1 - расчет к h_start
                            (по умолчанию - 1)
    :param h_start: 'top' или 'bottom' (по умолчанию top)
    :param mean_integral_error: True - среднеинтегральная ошибка,
                                False - относительная ошибка
                                (по умолчанию - False)
    """
    if equipment_data is None:
        equipment_data = {}

    if calc_options is None:
        calc_options = {'error_calc': True,
                        'save_results': None,
                        'plot_results': None}
    if heat_balance:
        raise ValueError('Учет теплопотерь не реализован')

    if not calc_options['scenario']:
        if sample:  # Если sample = True, значит расчет моделей будет на произвольном наборе параметров
            if pars_limits is None:  # Проверка, есть ли данные по pars_limits
                raise ValueError('Нет данных pars_limits')

            if calc_type.lower() == 'choke':
                choke = AutotestChoke(model_path,
                                      file_path,
                                      trajectory_data,
                                      ambient_temperature_data,
                                      fluid_data,
                                      data,
                                      equipment_data,
                                      hydr_corr_type)

                choke.sample_model(p1=p_atma,
                                   pars_limits=pars_limits,
                                   number_of_samples=number_of_samples,
                                   limited_pars=limited_pars,
                                   t_k=t_k,
                                   choke_data=data,
                                   fluid_data=fluid_data,
                                   model_path=model_path,
                                   flow_direction=flow_direction,
                                   file_path=file_path
                                   )

            if calc_type.lower() == 'pvt':  # тип расчета - PVT
                pvt = AutotestPVT(model_path,
                                  file_path,
                                  trajectory_data,
                                  ambient_temperature_data,
                                  fluid_data,
                                  data,
                                  equipment_data,
                                  hydr_corr_type)

                pvt.sample_model(pars_limits=pars_limits,
                                 limited_pars=limited_pars,
                                 number_of_samples=number_of_samples,
                                 pfl=p_atma,
                                 model_path=model_path,
                                 fluid_data=fluid_data,
                                 pipe_data=data,
                                 calc_type=calc_type,
                                 equipment_data=equipment_data,
                                 well_trajectory_data=trajectory_data,
                                 calc_options=calc_options,
                                 result_path=file_path,
                                 heat_balance=heat_balance,
                                 temperature_option=temperature_option,
                                 flow_direction=1,
                                 h_start='top',
                                 mean_integral_error=mean_integral_error,
                                 ambient_temperature_data=ambient_temperature_data)

            elif calc_type.lower() == 'esp':
                # if p_ann is None:
                #     raise ValueError('Не задано затрубное давление')
                esp = AutotestESP(model_path,
                                  file_path,
                                  trajectory_data,
                                  ambient_temperature_data,
                                  fluid_data,
                                  data,
                                  equipment_data,
                                  hydr_corr_type)

                esp.sample_model(pars_limits=pars_limits,
                                 limited_pars=limited_pars,
                                 number_of_samples=number_of_samples,
                                 pfl=p_atma,
                                 model_path=model_path,
                                 fluid_data=fluid_data,
                                 pipe_data=data,
                                 esp_data=esp_data,
                                 esp_id=esp_id,
                                 stage_by_stage=stage_by_stage,
                                 freq_q_ag=freq,
                                 calc_type=calc_type,
                                 equipment_data=equipment_data,
                                 well_trajectory_data=trajectory_data,
                                 calc_options=calc_options,
                                 result_path=file_path,
                                 heat_balance=heat_balance,
                                 temperature_option=temperature_option,
                                 flow_direction=1,
                                 h_start='top',
                                 mean_integral_error=mean_integral_error,
                                 ambient_temperature_data=ambient_temperature_data,
                                 p_ann=p_ann)

            elif calc_type.lower() == 'pipe':
                pipe = AutotestPipe(model_path,
                                    file_path,
                                    trajectory_data,
                                    ambient_temperature_data,
                                    fluid_data,
                                    data,
                                    equipment_data,
                                    hydr_corr_type)

                pipe.sample_model(pars_limits=pars_limits,
                                  limited_pars=limited_pars,
                                  number_of_samples=number_of_samples,
                                  pfl=p_atma,
                                  model_path=model_path,
                                  fluid_data=fluid_data,
                                  pipe_data=data,
                                  calc_type=calc_type,
                                  equipment_data=equipment_data,
                                  well_trajectory_data=trajectory_data,
                                  calc_options=calc_options,
                                  result_path=file_path,
                                  heat_balance=heat_balance,
                                  temperature_option=temperature_option,
                                  flow_direction=1,
                                  h_start='top',
                                  mean_integral_error=mean_integral_error,
                                  ambient_temperature_data=ambient_temperature_data)

            elif calc_type.lower() == 'well':  # в тип расчета well будет включен также и ЭЦН (пока только газлифт)
                if "gl_system" in equipment_data:
                    gaslift = AutotestGasLift(model_path,
                                              file_path,
                                              trajectory_data,
                                              ambient_temperature_data,
                                              fluid_data,
                                              data,
                                              equipment_data,
                                              hydr_corr_type)

                    gaslift.sample_model(pars_limits=pars_limits,
                                         limited_pars=limited_pars,
                                         number_of_samples=number_of_samples,
                                         pfl=p_atma,
                                         model_path=model_path,
                                         fluid_data=fluid_data,
                                         pipe_data=data,
                                         equipment_data=equipment_data,
                                         well_trajectory_data=trajectory_data,
                                         calc_type=calc_type,
                                         result_path=file_path,
                                         freq_q_ag=q_gas_inj,
                                         calc_options=calc_options,
                                         temperature_option=temperature_option,
                                         heat_balance=heat_balance,
                                         ambient_temperature_data=ambient_temperature_data,
                                         p_gas_inj=p_gas_inj)

                elif 'esp_system' in equipment_data:
                    if equipment_data['esp_system']['esp_electric_system'] is None:
                        raise ValueError('Для расчета скважины с ЭЦН нужны данные по электрической системе насоса '
                                         '(esp_electric_system)')
                    if p_ann is None:
                        raise ValueError('Нужно задать затрубное давление p_ann')

                    esp = AutotestESP(model_path,
                                      file_path,
                                      trajectory_data,
                                      ambient_temperature_data,
                                      fluid_data,
                                      data,
                                      equipment_data,
                                      hydr_corr_type)

                    esp.sample_model(pars_limits=pars_limits,
                                     limited_pars=limited_pars,
                                     number_of_samples=number_of_samples,
                                     pfl=p_atma,
                                     model_path=model_path,
                                     fluid_data=fluid_data,
                                     pipe_data=data,
                                     esp_data=esp_data,
                                     esp_id=esp_id,
                                     stage_by_stage=stage_by_stage,
                                     freq_q_ag=freq,
                                     calc_type=calc_type,
                                     equipment_data=equipment_data,
                                     well_trajectory_data=trajectory_data,
                                     calc_options=calc_options,
                                     result_path=file_path,
                                     heat_balance=heat_balance,
                                     temperature_option=temperature_option,
                                     flow_direction=1,
                                     h_start='top',
                                     mean_integral_error=mean_integral_error,
                                     ambient_temperature_data=ambient_temperature_data,
                                     p_ann=p_ann)

        else:  # если Sample = False, значит посчитается единичный расчет main по аналогичному алгоритму
            if calc_type.lower() == 'pvt':
                pvt = AutotestPVT(model_path,
                                  file_path,
                                  trajectory_data,
                                  ambient_temperature_data,
                                  fluid_data,
                                  data,
                                  equipment_data,
                                  hydr_corr_type)

                pvt.main(pfl=p_atma,
                         model_path=model_path,
                         fluid_data=fluid_data,
                         pipe_data=data,
                         calc_type=calc_type,
                         equipment_data=equipment_data,
                         well_trajectory_data=trajectory_data,
                         calc_options=calc_options,
                         result_path=file_path,
                         heat_balance=heat_balance,
                         temperature_option=temperature_option,
                         flow_direction=1,
                         h_start='top',
                         mean_integral_error=mean_integral_error,
                         ambient_temperature_data=ambient_temperature_data)

            elif calc_type.lower() == 'pipe':
                pipe = AutotestPipe(model_path,
                                    file_path,
                                    trajectory_data,
                                    ambient_temperature_data,
                                    fluid_data,
                                    data,
                                    equipment_data,
                                    hydr_corr_type)

                pipe.main(pfl=p_atma,
                          model_path=model_path,
                          fluid_data=fluid_data,
                          pipe_data=data,
                          calc_type=calc_type,
                          equipment_data=equipment_data,
                          well_trajectory_data=trajectory_data,
                          calc_options=calc_options,
                          result_path=file_path,
                          heat_balance=heat_balance,
                          temperature_option=temperature_option,
                          flow_direction=1,
                          h_start='top',
                          mean_integral_error=mean_integral_error,
                          ambient_temperature_data=ambient_temperature_data)

            elif calc_type.lower() == 'esp':
                esp = AutotestESP(model_path,
                                  file_path,
                                  trajectory_data,
                                  ambient_temperature_data,
                                  fluid_data,
                                  data,
                                  equipment_data,
                                  hydr_corr_type)

                esp.main(pfl=p_atma,
                         model_path=model_path,
                         fluid_data=fluid_data,
                         pipe_data=data,
                         esp_data=esp_data,
                         esp_id=esp_id,
                         freq=freq,
                         stage_by_stage=stage_by_stage,
                         calc_type=calc_type,
                         equipment_data=equipment_data,
                         well_trajectory_data=trajectory_data,
                         calc_options=calc_options,
                         result_path=file_path,
                         heat_balance=heat_balance,
                         temperature_option=temperature_option,
                         flow_direction=1,
                         h_start='top',
                         mean_integral_error=mean_integral_error,
                         ambient_temperature_data=ambient_temperature_data,
                         p_ann=p_ann)

            elif calc_type.lower() == 'well':
                if "gl_system" in equipment_data:
                    gaslift = AutotestGasLift(model_path,
                                              file_path,
                                              trajectory_data,
                                              ambient_temperature_data,
                                              fluid_data,
                                              data,
                                              equipment_data,
                                              hydr_corr_type)

                    gaslift.main(pfl=p_atma,
                                 result_path=file_path,
                                 model_path=model_path,
                                 fluid_data=fluid_data,
                                 pipe_data=data,
                                 equipment_data=equipment_data,
                                 well_trajectory_data=trajectory_data,
                                 calc_type=calc_type,
                                 freq=q_gas_inj,
                                 calc_options=calc_options,
                                 temperature_option=temperature_option,
                                 heat_balance=heat_balance,
                                 ambient_temperature_data=ambient_temperature_data,
                                 p_gas_inj=p_gas_inj)

                elif 'esp_system' in equipment_data:
                    if equipment_data['esp_system']['esp_electric_system'] is None:
                        raise ValueError('Для расчета скважины с ЭЦН нужны данные по электрической системе насоса '
                                         '(esp_electric_system)')
                    if p_ann is None:
                        raise ValueError('Нужно задать затрубное давление p_ann')

                    esp = AutotestESP(model_path,
                                      file_path,
                                      trajectory_data,
                                      ambient_temperature_data,
                                      fluid_data,
                                      data,
                                      equipment_data,
                                      hydr_corr_type)

                    esp.main(pfl=p_atma,
                             model_path=model_path,
                             fluid_data=fluid_data,
                             pipe_data=data,
                             esp_data=esp_data,
                             esp_id=esp_id,
                             freq=freq,
                             stage_by_stage=stage_by_stage,
                             calc_type=calc_type,
                             equipment_data=equipment_data,
                             well_trajectory_data=trajectory_data,
                             calc_options=calc_options,
                             result_path=file_path,
                             heat_balance=heat_balance,
                             temperature_option=temperature_option,
                             flow_direction=1,
                             h_start='top',
                             mean_integral_error=mean_integral_error,
                             ambient_temperature_data=ambient_temperature_data,
                             p_ann=p_ann)

    else:
        app = xw.App(visible=False)
        book = xw.Book()
        sheet = book.sheets[0]
        sheet.name = 'Сценарий 1'
        counter = 2
        if calc_type == 'pipe':
            trajectory_dict = {'vertical': {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
                                                                         data=[[float(0), float(0)],
                                                                               [float(1800), float(1800)]])},
                               'horizontal': {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
                                                                           data=[[float(0), float(0)],
                                                                                 [float(1800), float(0.01)]])},
                               '30degree': {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
                                                                         data=[[float(0), float(0)],
                                                                               [float(1800), float(1558)]])},
                               '60degree': {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
                                                                         data=[[float(0), float(0)],
                                                                               [float(1800), float(900)]])}}
            pars_limits_dict = {'water': {'wct': [0.99, 0.99],
                                          'q_fluid': [10 / 86400, 1000 / 86400],
                                          'rp': [0.0, 0.0],
                                          'd': [0.06, 0.2],
                                          't_res': [280.15, 380.15],
                                          'gamma_oil': [0.65, 0.85]},
                                'degas_oil': {'wct': [0.01, 0.01],
                                              'q_fluid': [10 / 86400, 1000 / 86400],
                                              'rp': [0.0, 0.0],
                                              'd': [0.06, 0.2],
                                              't_res': [280.15, 380.15],
                                              'gamma_oil': [0.65, 0.85]},
                                'gas_oil': {'wct': [0.0, 0.0],
                                            'q_fluid': [10 / 86400, 1000 / 86400],
                                            'rp': [10, 1000],
                                            'd': [0.06, 0.2],
                                            't_res': [280.15, 380.15],
                                            'gamma_oil': [0.65, 0.85]},
                                'multiphase': {'wct': [0.0, 0.99],
                                               'q_fluid': [10 / 86400, 1000 / 86400],
                                               'rp': [10, 1000],
                                               'd': [0.06, 0.1],
                                               't_res': [280.15, 380.15],
                                               'gamma_oil': [0.65, 0.85]}}

            for well_type, incl in trajectory_dict.items():
                for fluid_type, pars_limits_new in pars_limits_dict.items():
                    print(f"Сценарий {counter - 1}, тип скважины {well_type}, тип флюида {fluid_type}")
                    pipe = AutotestPipe(model_path,
                                        file_path,
                                        incl,
                                        ambient_temperature_data,
                                        fluid_data,
                                        data,
                                        equipment_data,
                                        hydr_corr_type)

                    scenario = pipe.sample_model(pars_limits=pars_limits_new,
                                                 limited_pars=limited_pars,
                                                 number_of_samples=number_of_samples,
                                                 pfl=p_atma,
                                                 model_path=model_path,
                                                 fluid_data=fluid_data,
                                                 pipe_data=data,
                                                 calc_type=calc_type,
                                                 equipment_data=equipment_data,
                                                 well_trajectory_data=incl,
                                                 calc_options=calc_options,
                                                 result_path=file_path,
                                                 heat_balance=heat_balance,
                                                 temperature_option=temperature_option,
                                                 flow_direction=1,
                                                 h_start='top',
                                                 mean_integral_error=mean_integral_error,
                                                 ambient_temperature_data=ambient_temperature_data)

                    # sheet = book.sheets[0]
                    sheet['A1'].value = f"Сценарий {counter - 1}, тип скважины {well_type}, тип флюида {fluid_type}"
                    sheet['A2'].value = scenario
                    book.sheets.add('Сценарий' + str(counter))
                    sheet = book.sheets['Сценарий' + str(counter)]
                    counter += 1
        elif calc_type == 'esp':
            pars_limits = {'water': {'wct': [0.99, 0.99],
                                     'q_fluid': None,
                                     'rp': [0.0, 0.0],
                                     'freq_q_ag': [40, 60]},
                           'degas_oil': {'wct': [0.0, 0.0],
                                         'q_fluid': None,
                                         'rp': [0.0, 0.0],
                                         'freq_q_ag': [40, 60]},
                           'gas_oil': {'wct': [0.0, 0.0],
                                       'q_fluid': None,
                                       'rp': [0, 50],
                                       'freq_q_ag': [40, 60]},
                           'multiphase': {'wct': [0.0, 0.99],
                                          'q_fluid': None,
                                          'rp': [0, 50],
                                          'freq_q_ag': [40, 60]}}
            esp_id_dict = {'id_50': 1016,  # Дебит 50м3/сут
                           'id_150': 362,  # Дебит 150м3/сут
                           'id_300': 1000}  # Дебит 300м3/сут
            for key, id in esp_id_dict.items():
                for fluid_type, pars_limits_new in pars_limits.items():
                    if id == 1016:
                        pars_limits[fluid_type]['q_fluid'] = [40 / 86400, 60 / 86400]
                        equipment_data['esp_system']['esp']['stages'] = 380
                    elif id == 362:
                        pars_limits[fluid_type]['q_fluid'] = [140 / 86400, 160 / 86400]
                        equipment_data['esp_system']['esp']['stages'] = 220
                    elif id == 1000:
                        pars_limits[fluid_type]['q_fluid'] = [290 / 86400, 310 / 86400]
                        equipment_data['esp_system']['esp']['stages'] = 180
                    print(f"Сценарий {counter - 1}, тип насоса {id}, тип флюида {fluid_type}")
                    esp = AutotestESP(model_path,
                                      file_path,
                                      trajectory_data,
                                      ambient_temperature_data,
                                      fluid_data,
                                      data,
                                      equipment_data,
                                      hydr_corr_type)

                    scenario = esp.sample_model(pars_limits=pars_limits_new,
                                                limited_pars=limited_pars,
                                                number_of_samples=number_of_samples,
                                                pfl=p_atma,
                                                model_path=model_path,
                                                fluid_data=fluid_data,
                                                pipe_data=data,
                                                esp_data=esp_data,
                                                esp_id=id,
                                                stage_by_stage=stage_by_stage,
                                                freq_q_ag=freq,
                                                calc_type=calc_type,
                                                equipment_data=equipment_data,
                                                well_trajectory_data=trajectory_data,
                                                calc_options=calc_options,
                                                result_path=file_path,
                                                heat_balance=heat_balance,
                                                temperature_option=temperature_option,
                                                flow_direction=1,
                                                h_start='top',
                                                mean_integral_error=mean_integral_error,
                                                ambient_temperature_data=ambient_temperature_data,
                                                p_ann=p_ann)
                    sheet['A1'].value = f"Сценарий {counter - 1}, id насоса {id}, тип флюида {fluid_type}"
                    sheet['A2'].value = scenario
                    book.sheets.add('Сценарий' + str(counter))
                    sheet = book.sheets['Сценарий' + str(counter)]
                    counter += 1
        elif calc_type == 'well':
            if 'esp_system' in equipment_data:

                if equipment_data['esp_system']['esp_electric_system'] is None:
                    raise ValueError('Для расчета скважины с ЭЦН нужны данные по электрической системе насоса '
                                     '(esp_electric_system)')
                if p_ann is None:
                    raise ValueError('Нужно задать затрубное давление p_ann')

                pars_limits = {'water': {'wct': [0.99, 0.99],
                                         'q_fluid': None,
                                         'rp': [0.0, 0.0],
                                         'freq_q_ag': [40, 60]},
                               'degas_oil': {'wct': [0.0, 0.0],
                                             'q_fluid': None,
                                             'rp': [0.0, 0.0],
                                             'freq_q_ag': [40, 60]},
                               'gas_oil': {'wct': [0.0, 0.0],
                                           'q_fluid': None,
                                           'rp': [0, 50],
                                           'freq_q_ag': [40, 60]},
                               'multiphase': {'wct': [0.0, 0.99],
                                              'q_fluid': None,
                                              'rp': [0, 50],
                                              'freq_q_ag': [40, 60]}}
                esp_id_dict = {'id_50': 1016,  # Дебит 50м3/сут
                               'id_150': 362,  # Дебит 150м3/сут
                               'id_300': 1000}  # Дебит 300м3/сут
                for key, id in esp_id_dict.items():
                    for fluid_type, pars_limits_new in pars_limits.items():
                        if id == 1016:
                            pars_limits[fluid_type]['q_fluid'] = [40 / 86400, 60 / 86400]
                            equipment_data['esp_system']['esp']['stages'] = 380
                        elif id == 362:
                            pars_limits[fluid_type]['q_fluid'] = [140 / 86400, 160 / 86400]
                            equipment_data['esp_system']['esp']['stages'] = 220
                        elif id == 1000:
                            pars_limits[fluid_type]['q_fluid'] = [290 / 86400, 310 / 86400]
                            equipment_data['esp_system']['esp']['stages'] = 180
                        print(f"Сценарий {counter - 1}, тип насоса {id}, тип флюида {fluid_type}")
                        esp = AutotestESP(model_path,
                                          file_path,
                                          trajectory_data,
                                          ambient_temperature_data,
                                          fluid_data,
                                          data,
                                          equipment_data,
                                          hydr_corr_type)

                        scenario = esp.sample_model(pars_limits=pars_limits_new,
                                                    limited_pars=limited_pars,
                                                    number_of_samples=number_of_samples,
                                                    pfl=p_atma,
                                                    model_path=model_path,
                                                    fluid_data=fluid_data,
                                                    pipe_data=data,
                                                    esp_data=esp_data,
                                                    esp_id=esp_id,
                                                    stage_by_stage=stage_by_stage,
                                                    freq_q_ag=freq,
                                                    calc_type=calc_type,
                                                    equipment_data=equipment_data,
                                                    well_trajectory_data=trajectory_data,
                                                    calc_options=calc_options,
                                                    result_path=file_path,
                                                    heat_balance=heat_balance,
                                                    temperature_option=temperature_option,
                                                    flow_direction=1,
                                                    h_start='top',
                                                    mean_integral_error=mean_integral_error,
                                                    ambient_temperature_data=ambient_temperature_data,
                                                    p_ann=p_ann)

                        sheet['A1'].value = f"Сценарий {counter - 1}, id насоса {id}, тип флюида {fluid_type}"
                        sheet['A2'].value = scenario
                        book.sheets.add('Сценарий' + str(counter))
                        sheet = book.sheets['Сценарий' + str(counter)]
                        counter += 1
        elif calc_type == 'pvt':
            if calc_type.lower() == 'pvt':  # тип расчета - PVT
                pars_limits_dict = {'water': {'wct': [0.99, 0.99],
                                              'q_fluid': [10 / 86400, 1000 / 86400],
                                              'rp': [0.0, 0.0],
                                              'd': [0.06, 0.2],
                                              't_res': [280.15, 380.15],
                                              'gamma_oil': [0.65, 0.85]},
                                    'degas_oil': {'wct': [0.01, 0.01],
                                                  'q_fluid': [10 / 86400, 1000 / 86400],
                                                  'rp': [0.0, 0.0],
                                                  'd': [0.06, 0.2],
                                                  't_res': [280.15, 380.15],
                                                  'gamma_oil': [0.65, 0.85]},
                                    'gas_oil': {'wct': [0.0, 0.0],
                                                'q_fluid': [10 / 86400, 1000 / 86400],
                                                'rp': [10, 1000],
                                                'd': [0.06, 0.2],
                                                't_res': [280.15, 380.15],
                                                'gamma_oil': [0.65, 0.85]},
                                    'multiphase': {'wct': [0.0, 0.99],
                                                   'q_fluid': [10 / 86400, 1000 / 86400],
                                                   'rp': [10, 1000],
                                                   'd': [0.06, 0.1],
                                                   't_res': [280.15, 380.15],
                                                   'gamma_oil': [0.65, 0.85]}}

                for fluid_type, pars_limits_new in pars_limits_dict.items():
                    print(f"Сценарий {counter - 1}, тип флюида {fluid_type}")
                    pvt = AutotestPVT(model_path,
                                      file_path,
                                      trajectory_data,
                                      ambient_temperature_data,
                                      fluid_data,
                                      data,
                                      equipment_data,
                                      hydr_corr_type)

                    scenario = pvt.sample_model(pars_limits=pars_limits_new,
                                                limited_pars=limited_pars,
                                                number_of_samples=number_of_samples,
                                                pfl=p_atma,
                                                model_path=model_path,
                                                fluid_data=fluid_data,
                                                pipe_data=data,
                                                calc_type=calc_type,
                                                equipment_data=equipment_data,
                                                well_trajectory_data=trajectory_data,
                                                calc_options=calc_options,
                                                result_path=file_path,
                                                heat_balance=heat_balance,
                                                temperature_option=temperature_option,
                                                flow_direction=1,
                                                h_start='top',
                                                mean_integral_error=mean_integral_error,
                                                ambient_temperature_data=ambient_temperature_data)
                    sheet['A1'].value = f"Сценарий {counter - 1}, тип флюида {fluid_type}"
                    sheet['A2'].value = scenario
                    book.sheets.add('Сценарий' + str(counter))
                    sheet = book.sheets['Сценарий' + str(counter)]
                    counter += 1
        book.save(file_path)
        book.close()
        app.quit()
