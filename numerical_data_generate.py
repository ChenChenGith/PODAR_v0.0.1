# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/11/28
# @Author  : Chen Chen (Tsinghua Univ.)
# @FileName: experiment_1step.py
# =====================================

from math import pi, cos, sin, sqrt
import pandas as pd
import pickle
from PODAR import Vehicles
import matplotlib.pyplot as plt
import numpy as np
import copy


def experiment_1(ego_state=0):
    '''
    conflicts from different angles and types and speed
    '''
    R = [15., 25., 35.]  # km/h
    CENTER = ['current', 'future'][0]
    ANGLE = [0., 45., 90., 135., 180., 225., 315]  # degree

    V_EGO = [0., 30.,]  # km/h
    M_EGO = 1.8  # t
    S_EGO = 0.
    V_CAR = [10., 30., 45.]  # km/h
    M_CAR = 1.8  # t
    S_CAR = 1.  # 损伤敏感性
    V_TRU = [30., 45.]  # km/h
    M_TRU = 4.2  # t
    S_TRU = 1.
    V_BIC = [5., 10.]  # km/h
    M_BIC = 0.09  # t
    S_BIC = 100.
    V_PED = [2.5, 5.]  # km/h
    M_PED = 0.06  # t
    S_PED = 100.
    EGO_Y = [0., -25.]

    vehicles = Vehicles()
    vehicles.set_ego(type='car', x0=0., y0=EGO_Y[ego_state], speed=V_EGO[ego_state] / 3.6, phi0=pi / 2)
    for r in R:
        for angle in ANGLE:
            angle = angle / 180 * pi
            a_ = pi + angle if angle < 0 else angle - pi
            for v_i in V_CAR:
                vehicles.add_obj(type='car', x0=r * cos(angle), y0=r * sin(angle), speed=v_i / 3.6, phi0=a_)
            for v_i in V_TRU:
                vehicles.add_obj(type='tru', x0=r * cos(angle), y0=r * sin(angle), speed=v_i/ 3.6, phi0=a_)
            # if angle * 180 / pi > 180: continue
            for v_i in V_BIC:
                vehicles.add_obj(type='bic', x0=r * cos(angle), y0=r * sin(angle), speed=v_i / 3.6, phi0=a_)
            for v_i in V_PED:
                vehicles.add_obj(type='ped', x0=r * cos(angle), y0=r * sin(angle), speed=v_i / 3.6, phi0=a_)

    vehicles.estimate_risk()
    res = vehicles.get_risk_in_stru()
    results = pd.DataFrame(res, columns=['ego_speed', 'r', 'type', 'phi', 'phi_de', 'x', 'y', 'ov_speed', 'ov_speed_km', 'risk', 'risk_curve', 'collided', 'rc'])
    with open(r'numerical_data/' + str(V_EGO[ego_state]) + r'kmph/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    with open(r'numerical_data/'  + str(V_EGO[ego_state]) + r'kmph/vehs.pkl', 'wb') as f:
        pickle.dump(vehicles, f)
    print('e1 saved')

def experiment_2():
    '''
    when the host vehicle is stationary
    '''
    vehicles_1 = Vehicles()
    vehicles_1.set_ego(type='car', x0=0., y0=0., speed=0 / 3.6, phi0=pi / 2)

    vehicles_1.add_obj(type='car', x0=3.5, y0=-25., speed=30 / 3.6, phi0=pi / 2)
    vehicles_1.add_obj(type='car', x0=3.5, y0=-10., speed=30 / 3.6, phi0=pi / 2)
    vehicles_1.add_obj(type='car', x0=3.5, y0=0., speed=30 / 3.6, phi0=pi / 2)
    vehicles_1.add_obj(type='car', x0=3.5, y0=10., speed=30 / 3.6, phi0=pi / 2)
    vehicles_1.add_obj(type='car', x0=3.5, y0=25., speed=30 / 3.6, phi0=pi / 2)

    vehicles_1.estimate_risk()

    init_pos = [[0., 0.], [3.5, -25]]
    risks = []
    x1, y1, x2, y2 = [], [], [], []
    # plt.figure(figsize=(5, 20))
    for t in np.arange(0, 6, 0.1):
        vehicles = Vehicles()
        vehicles.set_ego(type='car', x0=init_pos[0][0], y0=init_pos[0][1], speed=0 / 3.6, phi0=pi / 2)
        vehicles.add_obj(type='car', x0=init_pos[1][0], y0=init_pos[1][1], speed=30 / 3.6, phi0=pi / 2)
        init_pos[0][1] += 0.1 * 0 / 3.6
        init_pos[1][1] += 0.1 * 30 / 3.6

        risk, _, _ = vehicles.estimate_risk()

        risks.append(risk)
        # plt.scatter(init_pos[0][0], init_pos[0][1], c='r')
        # plt.scatter(init_pos[1][0], init_pos[1][1], c='g')
        # plt.text(init_pos[0][0], init_pos[0][1], '{:.3f}'.format(risk))
        # x1.append(init_pos[0][0])
        # y1.append(init_pos[0][1])
        x2.append(init_pos[1][0])
        y2.append(init_pos[1][1])
    # plt.scatter(x1, y1, c='r')
    # plt.scatter(x2, y2, c='g')
    # plt.text()
    # plt.axis('equal')
    # # plt.show()
    # plt.plot(np.array(risks) * 2 + 5, y1)
    # print(vehicles.obj[0].delta_v)
    # plt.show()

    with open(r'numerical_data/'  + 'static/vehs.pkl', 'wb') as f:
        pickle.dump((vehicles_1, risks, x2, y2), f)

def experiment_3():
    '''
    snap-shot: car-following
    '''
    vehicles = Vehicles()
    vehicles.set_ego(type='car', x0=0., y0=0., speed=30. / 3.6, phi0=0)
    # front
    vehicles.add_obj(type='car', x0=10., y0=0., speed=45. / 3.6, phi0=0)
    vehicles.add_obj(type='car', x0=10., y0=0., speed=30. / 3.6, phi0=0)
    vehicles.add_obj(type='car', x0=10., y0=0., speed=20. / 3.6, phi0=0)
    vehicles.add_obj(type='car', x0=10., y0=0., speed=15. / 3.6, phi0=0)
    # rear
    vehicles.add_obj(type='car', x0=-10., y0=0., speed=15. / 3.6, phi0=0)
    vehicles.add_obj(type='car', x0=-10., y0=0., speed=25. / 3.6, phi0=0)
    vehicles.add_obj(type='car', x0=-10., y0=0., speed=30. / 3.6, phi0=0)
    vehicles.add_obj(type='car', x0=-10., y0=0., speed=45. / 3.6, phi0=0)
    #front speciall: same lane
    # vehicles.add_obj(type='car', x0=0., y0=15., speed=45. / 3.6, phi0=pi / 2)

    vehicles.estimate_risk()
    with open(r'numerical_data/'  + 'car_following/vehs1.pkl', 'wb') as f:
        pickle.dump(vehicles, f)

def experiment_4():
    """Car following and compare with TTC, DRAC and SF
    """
    init_pos = [[0., 0.], [15.5, 0.]]  # ego pos, ov pos  v=15: collided,  v=20: no collision
    init_speed = [10, 5]  # km/h
    min_a = -2.5
    ae = np.arange(0,min_a,-0.1).tolist() + [min_a] * 0 + np.arange(min_a, 0, 0.1).tolist() + [0] * 11 + np.arange(0, 2, 0.1).tolist() + [2] * 41
    aov = [0] * 55 + np.arange(0, 1, 0.1).tolist() + [1] * 45
    a = [ae, aov]  # ego acce, ov acce
    risks, ttc, drac, sf, drf = [], [], [], [], []
    colliede = []
    x1, y1, x2, y2 = [], [], [], []
    v, d, ve, vov = [], [], [], []
    media_variable = {}
    for i in range(0,110):      
        vehicles = Vehicles()
        # vehicles.set_ego(type='car', x0=init_pos[0][0], y0=init_pos[0][1], speed=init_speed[0], a0=0, phi0=0)
        # vehicles.add_obj(type='car', x0=init_pos[1][0], y0=init_pos[1][1], speed=init_speed[1], a0=0, phi0=0)
        vehicles.set_ego(type='car', x0=init_pos[0][0], y0=init_pos[0][1], speed=init_speed[0], a0=a[0][i], phi0=0)
        vehicles.add_obj(type='car', x0=init_pos[1][0], y0=init_pos[1][1], speed=init_speed[1], a0=a[1][i], phi0=0)
        
        risk, _, rc = vehicles.estimate_risk()
        risks.append(risk)
        ttc.append(1 / ((init_pos[1][0] - init_pos[0][0] - 4.8) / (init_speed[0] - init_speed[1])))
        drac.append((init_speed[0] - init_speed[1]) ** 2 / abs(init_pos[1][0] - init_pos[0][0] - 4.8))
        sf.append(0.5 * vehicles.obj[0].mass * init_speed[1] ** 2 * 0.01 * (1000 / ((init_pos[1][0] - init_pos[0][0]) ** 2) + np.exp(init_speed[0] - init_speed[1])))  # 1/2mv^2/r^2 +  1/2mv^2(v_r*cos)
        drf.append(0.0064 * (init_pos[1][0] - init_pos[0][0] - init_speed[0] * 3.5) ** 2 * 3500)
        # print(sf[-1],
        #       0.5 * vehicles.obj[0].mass * init_speed[1] ** 2,
        #       1000 / ((init_pos[1][0] - init_pos[0][0]) ** 2),
        #       np.exp(init_speed[0] - init_speed[1]),
        #       init_speed[0] - init_speed[1])
        d.append(init_pos[1][0] - init_pos[0][0] - 4.8)
        v.append(init_speed[0] - init_speed[1])
        ve.append(init_speed[0])
        vov.append(init_speed[1])
        colliede.append(rc)
        
        x1.append(init_pos[0][0])
        x2.append(init_pos[1][0])
        
        init_speed[0] += 0.1 * a[0][i]
        init_speed[1] += 0.1 * a[1][i]
        
        init_pos[0][0] += 0.1 * init_speed[0]
        init_pos[1][0] += 0.1 * init_speed[1]
        
        media_variable[i] = vehicles

        veh_back = copy.deepcopy(vehicles)
        
    with open(r'numerical_data/' + r'car_following\vehs_dynamic_collision.pkl', 'wb') as f:
        pickle.dump((risks, x1, x2, (v, ve, vov), d, ttc, drac, sf, a, drf, media_variable, colliede), f)

    # ---------------
    init_pos = [[0., 0.], [25, 0.]]  # ego pos, ov pos  v=15: collided,  v=20: no collision
    init_speed = [10, 5]  # km/h
    min_a = -2.5
    ae = np.arange(0,min_a,-0.1).tolist() + [min_a] * 0 + np.arange(min_a, 0, 0.1).tolist() + [0] * 11 + np.arange(0, 2, 0.1).tolist() + [2] * 41
    aov = [0] * 55 + np.arange(0, 1, 0.1).tolist() + [1] * 45
    a = [ae, aov]  # ego acce, ov acce
    risks, ttc, drac, sf, drf = [], [], [], [], []
    colliede = []
    x1, y1, x2, y2 = [], [], [], []
    v, d, ve, vov = [], [], [], []
    media_variable = {}
    for i in range(0,110):      
        vehicles = Vehicles()
        # vehicles.set_ego(type='car', x0=init_pos[0][0], y0=init_pos[0][1], speed=init_speed[0], a0=0, phi0=0)
        # vehicles.add_obj(type='car', x0=init_pos[1][0], y0=init_pos[1][1], speed=init_speed[1], a0=0, phi0=0)
        vehicles.set_ego(type='car', x0=init_pos[0][0], y0=init_pos[0][1], speed=init_speed[0], a0=a[0][i], phi0=0)
        vehicles.add_obj(type='car', x0=init_pos[1][0], y0=init_pos[1][1], speed=init_speed[1], a0=a[1][i], phi0=0)
        
        risk, _, rc = vehicles.estimate_risk()
        risks.append(risk)
        ttc.append(1 / ((init_pos[1][0] - init_pos[0][0] - 4.8) / (init_speed[0] - init_speed[1])))
        drac.append((init_speed[0] - init_speed[1]) ** 2 / abs(init_pos[1][0] - init_pos[0][0] - 4.8))
        sf.append(0.5 * vehicles.obj[0].mass * init_speed[1] ** 2 * 0.01 * (1000 / ((init_pos[1][0] - init_pos[0][0]) ** 2) + np.exp(init_speed[0] - init_speed[1])))  # 1/2mv^2/r^2 +  1/2mv^2(v_r*cos)
        drf.append(0.0064 * (init_pos[1][0] - init_pos[0][0] - init_speed[0] * 3.5) ** 2 * 3500)
        # print(sf[-1],
        #       0.5 * vehicles.obj[0].mass * init_speed[1] ** 2,
        #       1000 / ((init_pos[1][0] - init_pos[0][0]) ** 2),
        #       np.exp(init_speed[0] - init_speed[1]),
        #       init_speed[0] - init_speed[1])
        d.append(init_pos[1][0] - init_pos[0][0] - 4.8)
        v.append(init_speed[0] - init_speed[1])
        ve.append(init_speed[0])
        vov.append(init_speed[1])
        colliede.append(rc)
        
        x1.append(init_pos[0][0])
        x2.append(init_pos[1][0])
        
        init_speed[0] += 0.1 * a[0][i]
        init_speed[1] += 0.1 * a[1][i]
        
        init_pos[0][0] += 0.1 * init_speed[0]
        init_pos[1][0] += 0.1 * init_speed[1]
        
        media_variable[i] = vehicles

        veh_back = copy.deepcopy(vehicles)
        
    with open(r'numerical_data/' + r'car_following\vehs_dynamic_collision_free.pkl', 'wb') as f:
        pickle.dump((risks, x1, x2, (v, ve, vov), d, ttc, drac, sf, a, drf, media_variable, colliede), f)

    return(ve, vov, risks, ae, aov)

def generate():
    experiment_1(ego_state=0)
    experiment_1(ego_state=1)
    experiment_2()
    experiment_3()
    experiment_4()

if __name__ == '__main__':
    experiment_1(ego_state=0)
    experiment_1(ego_state=1)
    experiment_2()
    experiment_3()
    risk = experiment_4()