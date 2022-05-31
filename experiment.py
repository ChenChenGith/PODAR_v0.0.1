import numpy as np
from PODAR import Vehicles
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from math import cos, sin, atan2, pi
from matplotlib import colors
import matplotlib
import seaborn as sns

c_cyc = mcolors.to_rgba_array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
sns.set_theme(style="darkgrid")
plt.rc('font',family='Times New Roman') 

cmap = plt.cm.jet
c1 = [0 / 255, 255 / 255, 0 / 255]
c2 = [255 / 255, 255 / 255, 0 / 255]
c3 = [255 / 255, 0 / 255, 0 / 255]
cmaplist = [c1, c2, c3]
mycmap_3type = cmap.from_list('Custom cmap', cmaplist, cmap.N)

def comparison_measures():
    """Car following and compare with TTC, DRAC and SF
    """
    init_pos = [[0., 0.], [20, 0.]]  # ego pos, ov pos  v=15: collided,  v=20: no collision
    init_speed = [10, 5]  # km/h
    min_a = -2.5
    ae = np.arange(0,min_a,-0.1).tolist() + [min_a] * 0 + np.arange(min_a, 0, 0.1).tolist() + [0] * 11 + np.arange(0, 2, 0.1).tolist() + [2] * 41
    aov = [0] * 55 + np.arange(0, 1, 0.1).tolist() + [1] * 45
    a = [ae, aov]  # ego acce, ov acce
    risks, ttc, drac, sf = [], [], [], []
    colliede = []
    x1, y1, x2, y2 = [], [], [], []
    v, d, ve, vov = [], [], [], []
    media_variable = {}
    for i in range(0,110):      
        vehicles = Vehicles()
        vehicles.set_ego(type='car', x0=init_pos[0][0], y0=init_pos[0][1], speed=init_speed[0], a0=a[0][i], phi0=0)
        vehicles.add_obj(type='car', x0=init_pos[1][0], y0=init_pos[1][1], speed=init_speed[1], a0=a[1][i], phi0=0)
        
        risk, _, rc = vehicles.estimate_risk()
        risks.append(risk)
        ttc.append(1 / ((init_pos[1][0] - init_pos[0][0] - 4.8) / (init_speed[0] - init_speed[1])))
        drac.append((init_speed[0] - init_speed[1]) ** 2 / abs(init_pos[1][0] - init_pos[0][0] - 4.8))
        sf.append(0.5 * vehicles.obj[0].mass * init_speed[1] ** 2 * 0.01 * (1000 / ((init_pos[1][0] - init_pos[0][0]) ** 2) + np.exp(init_speed[0] - init_speed[1])))  # 1/2mv^2/r^2 +  1/2mv^2(v_r*cos)
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
        
    risks, x1, x2, (v, ve, vov), d, ttc, drac, sf, a, media_variable, colliede

    fig = plt.figure(figsize=(15,9))
    grid = plt.GridSpec(2,3,wspace=0.45,hspace=0.3)
    ti = 11

    zero_rv_min = np.where(np.array(v)<0)[0].min() / 10
    zero_rv_max = np.where(np.array(v)<0)[0].max() / 10

    ax1 = plt.subplot(grid[1,1])
    plt.plot(np.arange(0, ti, 0.1), risks, label='PCD risk')
    plt.title('d) PCD risk',fontsize=18)
    plt.xlabel('Time [second]',fontsize=15)
    plt.plot([zero_rv_min, zero_rv_min], [0, 0.15], c='black', linestyle=':')
    plt.plot([zero_rv_max, zero_rv_max], [0, 0.35], c='black', linestyle=':')

    ax2 = plt.subplot(grid[0,2])
    ax2ln1 = plt.plot(np.arange(0, ti, 0.1), ttc, label='1/TTC')
    # plt.scatter(np.arange(0, ti, 0.1), ttc, label='1/TTC')
    plt.ylabel('1/TTC [1/second]',fontsize=15)
    plt.xlabel('Time [second]',fontsize=15)
    plt.title('e) 1/TTC and DRAC',fontsize=18)
    plt.plot([0, ti], [0, 0], linestyle='--', c=list(c_cyc[0][:-1]) + [0.5])
    plt.plot([zero_rv_min, zero_rv_min], [-0.4, 0], c='black', linestyle=':')
    plt.plot([zero_rv_max, zero_rv_max], [-0.4, 0], c='black', linestyle=':')
    plt.text(0,0,'1/TTC=0', c=c_cyc[0])

    ax22 = ax2.twinx()
    ax2ln2 = plt.plot(np.arange(0, ti, 0.1), drac, label='DRAC', c=c_cyc[1])
    plt.plot([0, ti], [0, 0], linestyle='--', c=list(c_cyc[1][:-1]) + [0.5])
    plt.ylabel('DRAC [$m/s^2$]',fontsize=15)
    plt.legend(ax2ln1+ax2ln2, ['1/TTC', 'DRAC'])
    plt.text(0,0,'DRAC=0', c=c_cyc[1])
    plt.grid(None)

    ax3 = plt.subplot(grid[1,2])
    plt.plot(np.arange(0, ti, 0.1), sf, label='Safety Field')
    plt.title('f) Safety Field',fontsize=18)
    plt.xlabel('Time [second]',fontsize=15)
    plt.plot([zero_rv_min, zero_rv_min], [0, 3], c='black', linestyle=':')
    plt.plot([zero_rv_max, zero_rv_max], [0, 3], c='black', linestyle=':')

    ax4 = plt.subplot(grid[0,1])
    ax4ln1 = plt.plot(np.arange(0, ti, 0.1), v, label='Relative speed')
    plt.plot([0, ti], [0, 0], linestyle='--', c=list(c_cyc[0][:-1]) + [0.5])
    plt.text(0,0,'$\delta v=0$', c=c_cyc[0])
    plt.plot([zero_rv_min, zero_rv_min], [-2, 0], c='black', linestyle=':')
    plt.plot([zero_rv_max, zero_rv_max], [-2, 3.4], c='black', linestyle=':')
    plt.title('c) Relative speed/distance',fontsize=18)
    plt.ylabel('Relative speed [$m/s$]',fontsize=15)
    plt.xlabel('Time [second]',fontsize=15)

    ax44 = ax4.twinx()
    ax4ln2 = plt.plot(np.arange(0, ti, 0.1), d, label='Relative distance', c=c_cyc[1])
    plt.plot([0, ti], [0, 0], linestyle='--', c=list(c_cyc[1][:-1]) + [0.5])
    plt.text(0,0,'$\Delta d=0$', c=c_cyc[1])
    plt.ylabel('Relative distance [m]')
    plt.legend(ax4ln1+ax4ln2, ['Relative speed', 'Relative distance'])
    plt.grid(None)

    ax5 = plt.subplot(grid[0,0])
    plt.plot(np.arange(0, ti, 0.1), ve, label='Ego speed')
    # ax1 = plt.subplot(grid[1,3])
    plt.plot(np.arange(0, ti, 0.1), vov, label='OV distance')
    plt.title('a) Speed [$m/s$]',fontsize=18)
    plt.xlabel('Time [second]',fontsize=15)
    plt.legend()
    plt.plot([zero_rv_min, zero_rv_min], [3, 5], c='black', linestyle=':')
    plt.plot([zero_rv_max, zero_rv_max], [3, 8.4], c='black', linestyle=':')

    ax6 = plt.subplot(grid[1,0])
    plt.plot(np.arange(0, ti, 0.1), a[0][:ti*10], label='Acceleration')
    plt.plot(np.arange(0, ti, 0.1), a[1][:ti*10], label='Acceleration')
    plt.title('b) Acceleration [$m/s^2$]',fontsize=18)
    plt.legend()
    plt.xlabel('Time [second]',fontsize=15)
    
    plt.show()
    
def car_following():
    """side-by-side passing situation
    """
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
        x2.append(init_pos[1][0])
        y2.append(init_pos[1][1])

    vehs, dym_risks, obj_traj_x, obj_traj_y = vehicles_1, risks, x2, y2
    ego = vehs.ego
    risk_all = np.array([x.risk_curve for x in vehs.obj])
    risk_all = risk_all.reshape(155)
    x_all = np.concatenate([np.array(x.x_pred)+offset for (x, offset) in zip(vehs.obj, [0,1.5,3,4.5,6])])
    y_all = np.concatenate([x.y_pred for x in vehs.obj])
    risk = [x.risk for x in vehs.obj]
    dym_risks_2 = np.array(dym_risks)*2
    c_norm = colors.Normalize(vmin=dym_risks_2.min(), vmax=dym_risks_2.max(), clip=True)
    rcurve_color = mycmap_3type(c_norm(dym_risks_2))
    c_norm_1 = colors.Normalize(vmin=np.array(dym_risks).min(), vmax=np.array(dym_risks).max(), clip=True)

    fig = plt.figure(figsize=(18,9))
    grid = plt.GridSpec(2,3,wspace=0.3,hspace=0.3, )

    ax1 = plt.subplot(grid[:,0])
    # ego
    obj = ego
    w, l, h0 = obj.width, obj.length, obj.phi_pred[0]
    rr = np.sqrt(w**2 + l**2) / 2
    betha = atan2(w / 2, l / 2)
    x = obj.x_pred[0] - rr * cos(h0 + betha)
    y = obj.y_pred[0] - rr * sin(h0 + betha)
    rect = patches.Rectangle((x, y), l, w, angle=h0 / pi * 180, fill=True, edgecolor='#FFFFFF', facecolor='#33A1C9',linewidth=2, label='Ego Veh.')
    ax1.add_patch(rect)

    plot_sta(vehs.obj[0], risk_all, id_=0.0, color='#FF8000', risk=np.round(risk[0],3), label=True)
    plot_sta(vehs.obj[1], risk_all, id_=1.8, color='#FF8000', risk=np.round(risk[1],3), offset=1.5)
    plot_sta(vehs.obj[2], risk_all, id_=3.0, color='#FF8000', risk=np.round(risk[2],3), offset=3)
    plot_sta(vehs.obj[3], risk_all, id_=4.2, color='#FF8000', risk=np.round(risk[3],3), offset=4.5)
    plot_sta(vehs.obj[4], risk_all, id_=6.0, color='#FF8000', risk=np.round(risk[4],3), offset=6)

    ax1.scatter(x_all, y_all, s=40, c=risk_all, cmap=mycmap_3type, marker='^', zorder=10, label='Pred. Traj.')

    im1 = matplotlib.cm.ScalarMappable(cmap=mycmap_3type, norm=c_norm_1)
    cb = fig.colorbar(im1)
    cb.set_label('Risk', fontsize=15)
    cb.ax.tick_params(axis="y", labelsize=15)

    plt.plot(np.array(dym_risks) *2  + 26, np.array(obj_traj_y), c='#00FFFF', label='Driving Risks')
    plt.barh(obj_traj_y,np.array(dym_risks)*2,linewidth=1, height=1, left=[26.], color=rcurve_color)

    plt.plot([-1.75, -1.75], [-25, 40], c='gray', linewidth=1, dashes=(6, 9), label='Lane Marker') #linestyle='--',
    plt.plot([1.75, 1.75], [-25, 40], c='gray', linewidth=1, dashes=(6, 9)) #linestyle='--',
    plt.plot([10.75, 10.75], [-25, 40], c='gray', linewidth=1, dashes=(6, 9)) #linestyle='--',

    plt.axis('equal')
    # plt.xticks(fontsize=15)
    # ax1.tick_params(axis="x", labelsize=15)
    # plt.setp(ax1.get_xticklabels(), fontsize=15, fontweight="bold", horizontalalignment="left")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(np.linspace(-15,25,9))
    plt.xlim((-5, 30))
    plt.ylim((-30, 55))
    # plt.barh([-25, -10, 0, 10, 25], np.array([x.risk for x in vehs.obj]) * 20, linewidth=2, height=2, left=[3.], label='Overall risk', color=['#FFD700','#FF7D40','#FF8000','#FF7D40','#FFD700'])

    plt.legend(loc='upper right')
    plt.xlabel('x offset[m]',fontsize=15)
    plt.ylabel('y[m]',fontsize=15)
    ax1.set_title('a) Risks during driving',fontsize=20)

    ax2 = plt.subplot(grid[0,1])
    t = np.linspace(0,3,31)
    ax2lin1 = ax2.plot(t, vehs.obj[0].risk_curve, label='$t=0.0$')
    ax2lin2 = ax2.plot(t, vehs.obj[1].risk_curve, label='$t=1.8$')
    ax2lin3 = ax2.plot(t, vehs.obj[2].risk_curve, label='$t=3.0$')
    ax2lin4 = ax2.plot(t, vehs.obj[3].risk_curve, label='$t=4.2$')
    ax2lin5 = ax2.plot(t, vehs.obj[4].risk_curve, label='$t=6.0$')
    ax2.legend()
    ax2.set_title('b) Risk during prediction horizon',fontsize=20)
    ax2.tick_params(axis="x", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax2.set_xlabel('Prediction horizon[second]', fontsize=15)
    ax2.set_ylabel('Risk', fontsize=15)

    ax3 = plt.subplot(grid[1,1])
    ax3.plot(t, vehs.obj[0].dis_de_curve, label='$t=0.0$')
    ax3.plot(t, vehs.obj[1].dis_de_curve, label='$t=1.8$')
    ax3.plot(t, vehs.obj[2].dis_de_curve, label='$t=3.0$')
    ax3.plot(t, vehs.obj[3].dis_de_curve, label='$t=4.2$')
    ax3.plot(t, vehs.obj[4].dis_de_curve, label='$t=6.0$')
    ax3.legend()
    ax3.set_title('c) Distance attenuation curve',fontsize=20)
    ax3.tick_params(axis="x", labelsize=15)
    ax3.tick_params(axis="y", labelsize=15)
    ax3.set_xlabel('Prediction horizon[second]', fontsize=15)
    ax3.set_ylabel('$\omega$', fontsize=15)

    ax4 = plt.subplot(grid[0,2])
    ax4.plot(t, vehs.obj[0].damage, label='$t=0.0$')
    ax4.plot(t, vehs.obj[1].damage, label='$t=1.8$')
    ax4.plot(t, vehs.obj[2].damage, label='$t=3.0$')
    ax4.plot(t, vehs.obj[3].damage, label='$t=4.2$')
    ax4.plot(t, vehs.obj[4].damage, label='$t=6.0$')
    ax4.legend()
    ax4.set_title('d) Damage during prediction horizon',fontsize=20)
    ax4.tick_params(axis="x", labelsize=15)
    ax4.tick_params(axis="y", labelsize=15)
    ax4.set_xlabel('Prediction horizon[second]', fontsize=15)
    ax4.set_ylabel('Damage', fontsize=15)

    ax5 = plt.subplot(grid[1,2])
    ax5.plot(t, vehs.obj[0].weight_t, label='$t=0.0$')
    ax5.plot(t, vehs.obj[1].weight_t, label='$t=1.8$')
    ax5.plot(t, vehs.obj[2].weight_t, label='$t=3.0$')
    ax5.plot(t, vehs.obj[3].weight_t, label='$t=4.2$')
    ax5.plot(t, vehs.obj[4].weight_t, label='$t=6.0$')
    ax5.legend()
    ax5.set_title('e) Time attenuation curve',fontsize=20)
    ax5.tick_params(axis="x", labelsize=15)
    ax5.tick_params(axis="y", labelsize=15)
    ax5.set_xlabel('Prediction horizon[second]', fontsize=15)
    ax5.set_ylabel('$\omega$', fontsize=15)
    
    plt.show()

def plot_sta(obj, risk_all, id_, color, risk, offset=0, label=False):
    w, l, h0 = obj.width, obj.length, obj.phi_pred[0]
    rr = np.sqrt(w**2 + l**2) / 2
    betha = atan2(w / 2, l / 2)
    x = obj.x_pred[0] - rr * cos(h0 + betha) + offset
    y = obj.y_pred[0] - rr * sin(h0 + betha)
    if label:
        rect = patches.Rectangle((x, y), l, w, angle=h0 / pi * 180, fill=True, edgecolor='#FFFFFF', facecolor=color, linewidth=2, label='Other Veh.', alpha=0.6)
    else:
        rect = patches.Rectangle((x, y), l, w, angle=h0 / pi * 180, fill=True, edgecolor='#FFFFFF', facecolor=color, linewidth=2, alpha=0.6)
    # plt.text(x + 0.5, y, 'risk=' + str(risk),fontsize=15)
    plt.text(x + 0.5, y, '$R_{t=' + str(id_) + '}$=' + str(risk),fontsize=15)
    plt.gca().add_patch(rect)


if __name__ == '__main__':
    # comparison_measures()
    # car_following()

    vehicles_1 = Vehicles()
    vehicles_1.set_ego(type='car', x0=0., y0=0., speed=3.5, phi0=1.57)

    vehicles_1.add_obj(type='car', x0=0, y0=10.5, speed=5, phi0=1.57)
    vehicles_1.add_obj(type='car', x0=3.5, y0=0., speed=4, phi0=1.57)

    print(vehicles_1.estimate_risk())
    