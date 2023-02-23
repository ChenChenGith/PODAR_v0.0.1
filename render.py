import pickle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import cos, sin, pi
from typing import List, Tuple
from matplotlib import colors
from PODAR import Vehicles
import random

class Render(object):
    def __init__(self, xyticklim=None, global_norm=None, draw_map=True, grid=False, title=None, ana_mode=False):
        '''
        xytiklim:(xleft, xright, ybottom, ytop)
        '''
        self.fig = plt.figure(figsize=(8, 8))
        plt.ion()

        plt.rcParams['font.family'] = ['Times New Roman']
        plt.rcParams['font.size'] = 12 

        self.ax = self.fig.add_subplot()
        if not title == None: self.ax.set_title(title, loc='left')
        self.ax.axis("equal")
        if grid:
            self.ax.grid()
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        # self.ax.axis('off')

        self.ax_line = self.fig.add_axes([0.05, 0.05, 0.3, 0.3])
        # self.ax_line.set_title('risk')
        self.risk_collect = {'time': [], 'risks': [], 'rc': [[], []], 'vc': [[], []]}
        self.risk_line_handle, = self.ax_line.plot([], [], label='risk(red=coll. alert)')
        self.coll_scatter_handle = self.ax_line.scatter([], [], color='red', label='true coll.')
        self.pre_coll_scatter_handle = self.ax_line.scatter([], [], color='orange', label='pred. coll.')
        # self.ax_line.legend(loc='upper right',fontsize=8)

        self.ana_mode = ana_mode
        if ana_mode:
            self.info_collect = {'dis_weight': [], 'time_weight': [], 'delta_v': [], 'risk_curve': [], 'max_r_step': 0,
                                'damage': [],}

            self.ax_ana_1 = self.fig.add_axes([0.05, 0.4, 0.3, 0.3])
            self.dis_weight_handle, = self.ax_ana_1.plot([], [], label='$\omega_d$', c='red')
            self.max_r_handle_1 = self.ax_ana_1.scatter([], [], color='black', label='max risk step')
            self.time_weight_handle, = self.ax_ana_1.plot([], [], label='$\omega_t$', c='black')
            self.max_r_handle_11 = self.ax_ana_1.scatter([], [], color='black', label='max risk step')
            self.ax_ana_1.legend(loc='best')

            self.ax_ana_2 = self.fig.add_axes([0.65, 0.65, 0.3, 0.3])
            self.delta_v_handle, = self.ax_ana_2.plot([], [], label='delta_v pred', c='green')
            self.max_r_handle_2 = self.ax_ana_2.scatter([], [], color='black', label='max risk step')
            self.ax_ana_2.legend(loc='best')
            self.max_y, self.min_y = 0, 0

            self.ax_ana_3 = self.fig.add_axes([0.65, 0.05, 0.3, 0.3])
            self.risk_curve_handle, = self.ax_ana_3.plot([], [], label='risk pred', c='blue')
            self.max_r_handle_3 = self.ax_ana_3.scatter([], [], color='black', label='max risk step')
            # self.damage_curve_handle, = self.ax_ana_3.plot([], [], label='damage pred', c='black')
            # self.max_r_handle_33 = self.ax_ana_3.scatter([], [], color='black', label='max risk step')
            self.ax_ana_3.legend(loc='best')
            self.max_y_1, self.min_y_1 = 0, 0

            self.text_handle = self.ax_ana_3.text(self.max_y_1, self.min_y_1, '')

        self.square_length = 50  # 50
        self.extension = 60
        self.lane_width = 3.75
        self.light_line_width = 3
        self.dotted_line_style = '--'
        self.solid_line_style = '-'
        self.lane_number = 3

        self.last_step_vehs_rec = {}
        self.last_step_vehs_text = {}
        self.time_handle = None
        self.ego_veh_name = None

        self.v_norm = global_norm
        self.xyticklim = xyticklim
        self.draw_map = draw_map

        if draw_map:
            self._draw_map()
        self._set_ticks()
        self._set_colormap()

    def _set_colormap(self):
        cmap = plt.cm.jet
        c1 = [0 / 255, 255 / 255, 0 / 255]
        c2 = [255 / 255, 255 / 255, 0 / 255]
        c3 = [255 / 255, 0 / 255, 0 / 255]
        cmaplist = [c1, c2, c3]
        self.mycmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        if self.v_norm != None:
            self.c_norm = colors.Normalize(vmin=self.v_norm[0], vmax=self.v_norm[1], clip=True)
        else:
            self.c_norm = colors.Normalize(0, vmax=2, clip=True)

    def _is_in_risk_area(self, veh, dis=220):
        if self.ego_veh_name == None: return False
        if abs(self.ego_veh['x'] - veh['x']) + abs(self.ego_veh['y'] - veh['y']) < dis:
            if self.ego_veh['road'][0] != ':':
                if 120 < abs(self.ego_veh['phi'] - veh['phi']) < 240:
                    return False
                else:
                    return True
            else:
                return True
        else:
            return False

    def _is_in_plot_area(self, veh, tolerance=5):
        if self.xleft + tolerance < veh['x'] < self.xright - tolerance and \
                self.ybottom + tolerance < veh['y'] < self.ytop - tolerance:
            return True
        else:
            return False

    def _set_ticks(self):
        if not self.xyticklim == None:
            self.xleft, self.xright, self.ybottom, self.ytop = \
                self.xyticklim[0], self.xyticklim[1], self.xyticklim[2], self.xyticklim[3]
        else:
            self.xleft, self.xright, self.ybottom, self.ytop = \
                -self.square_length / 2 - self.extension, \
                self.square_length / 2 + self.extension, \
                -self.square_length / 2 - self.extension, \
                self.square_length / 2 + self.extension
        self.ax.set_xlim(self.xleft, self.xright)
        self.ax.set_ylim(self.ybottom, self.ytop)

    def _set_color(self, vehs_in_plot_area, vehs_in_risk_area):
        for veh in vehs_in_plot_area:
            if not veh['name'] == self.ego_veh_name:
                if veh in vehs_in_risk_area:
                    veh['edge_color'] = 'blue'
                    veh['face_color'] = self.mycmap(self.c_norm(veh['risk']))
                else:
                    veh['edge_color'] = 'black'
                    veh['face_color'] = 'white'
                    veh['risk'] = ''
            else:
                veh['edge_color'] = 'red'
                veh['face_color'] = 'white'

    def _draw_map(self):
        # ----------arrow--------------
        self.ax.arrow(self.lane_width / 2, -self.square_length / 2 - 10, 0, 5, color='orange')
        self.ax.arrow(self.lane_width / 2, -self.square_length / 2 - 10 + 5, -0.5, 0, color='orange', head_width=1)
        self.ax.arrow(self.lane_width * 1.5, -self.square_length / 2 - 10, 0, 4, color='orange', head_width=1)
        self.ax.arrow(self.lane_width * 2.5, -self.square_length / 2 - 10, 0, 5, color='orange')
        self.ax.arrow(self.lane_width * 2.5, -self.square_length / 2 - 10 + 5, 0.5, 0, color='orange', head_width=1)

        # ----------horizon--------------

        self.ax.plot([-self.square_length / 2 - self.extension, -self.square_length / 2], [0.3, 0.3], color='orange')
        self.ax.plot([-self.square_length / 2 - self.extension, -self.square_length / 2], [-0.3, -0.3], color='orange')
        self.ax.plot([self.square_length / 2 + self.extension, self.square_length / 2], [0.3, 0.3], color='orange')
        self.ax.plot([self.square_length / 2 + self.extension, self.square_length / 2], [-0.3, -0.3], color='orange')

        #
        for i in range(1, self.lane_number + 1):
            linestyle = self.dotted_line_style if i < self.lane_number else self.solid_line_style
            linewidth = 1 if i < self.lane_number else 2
            self.ax.plot([-self.square_length / 2 - self.extension, -self.square_length / 2],
                         [i * self.lane_width, i * self.lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([self.square_length / 2 + self.extension, self.square_length / 2],
                         [i * self.lane_width, i * self.lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([-self.square_length / 2 - self.extension, -self.square_length / 2],
                         [-i * self.lane_width, -i * self.lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([self.square_length / 2 + self.extension, self.square_length / 2],
                         [-i * self.lane_width, -i * self.lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)

        for i in range(4, 5 + 1):
            linestyle = self.dotted_line_style if i < 5 else self.solid_line_style
            linewidth = 1 if i < 5 else 2
            self.ax.plot([-self.square_length / 2 - self.extension, -self.square_length / 2],
                         [3 * self.lane_width + (i - 3) * 2, 3 * self.lane_width + (i - 3) * 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([self.square_length / 2 + self.extension, self.square_length / 2],
                         [3 * self.lane_width + (i - 3) * 2, 3 * self.lane_width + (i - 3) * 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([-self.square_length / 2 - self.extension, -self.square_length / 2],
                         [-3 * self.lane_width - (i - 3) * 2, -3 * self.lane_width - (i - 3) * 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([self.square_length / 2 + self.extension, self.square_length / 2],
                         [-3 * self.lane_width - (i - 3) * 2, -3 * self.lane_width - (i - 3) * 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

        # ----------vertical----------------
        self.ax.plot([0.3, 0.3], [-self.square_length / 2 - self.extension, -self.square_length / 2], color='orange')
        self.ax.plot([-0.3, -0.3], [-self.square_length / 2 - self.extension, -self.square_length / 2], color='orange')
        self.ax.plot([0.3, 0.3], [self.square_length / 2 + self.extension, self.square_length / 2], color='orange')
        self.ax.plot([-0.3, -0.3], [self.square_length / 2 + self.extension, self.square_length / 2], color='orange')

        #
        for i in range(1, self.lane_number + 1):
            linestyle = self.dotted_line_style if i < self.lane_number else self.solid_line_style
            linewidth = 1 if i < self.lane_number else 2
            self.ax.plot([i * self.lane_width, i * self.lane_width],
                         [-self.square_length / 2 - self.extension, -self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([i * self.lane_width, i * self.lane_width],
                         [self.square_length / 2 + self.extension, self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([-i * self.lane_width, -i * self.lane_width],
                         [-self.square_length / 2 - self.extension, -self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([-i * self.lane_width, -i * self.lane_width],
                         [self.square_length / 2 + self.extension, self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

        for i in range(4, 5 + 1):
            linestyle = self.dotted_line_style if i < 5 else self.solid_line_style
            linewidth = 1 if i < 5 else 2
            self.ax.plot([3 * self.lane_width + (i - 3) * 2, 3 * self.lane_width + (i - 3) * 2],
                         [-self.square_length / 2 - self.extension, -self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([3 * self.lane_width + (i - 3) * 2, 3 * self.lane_width + (i - 3) * 2],
                         [self.square_length / 2 + self.extension, self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([-3 * self.lane_width - (i - 3) * 2, -3 * self.lane_width - (i - 3) * 2],
                         [-self.square_length / 2 - self.extension, -self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            self.ax.plot([-3 * self.lane_width - (i - 3) * 2, -3 * self.lane_width - (i - 3) * 2],
                         [self.square_length / 2 + self.extension, self.square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

        # ----------Oblique--------------

        self.ax.plot([self.lane_number * self.lane_width + 4, self.square_length / 2],
                     [-self.square_length / 2, -self.lane_number * self.lane_width - 4],
                     color='black', linewidth=2)
        self.ax.plot([self.lane_number * self.lane_width + 4, self.square_length / 2],
                     [self.square_length / 2, self.lane_number * self.lane_width + 4],
                     color='black', linewidth=2)
        self.ax.plot([-self.lane_number * self.lane_width - 4, -self.square_length / 2],
                     [-self.square_length / 2, -self.lane_number * self.lane_width - 4],
                     color='black', linewidth=2)
        self.ax.plot([-self.lane_number * self.lane_width - 4, -self.square_length / 2],
                     [self.square_length / 2, self.lane_number * self.lane_width + 4],
                     color='black', linewidth=2)

        # ----------人行横道--------------
        jj = 3.5
        for ii in range(23):
            if ii <= 3:
                continue
            self.ax.add_patch(
                patches.Rectangle((-self.square_length / 2 + jj + ii * 1.6, -self.square_length / 2 + 0.5), 0.8, 4,
                                  color='lightgray', alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            self.ax.add_patch(
                patches.Rectangle((-self.square_length / 2 + jj + ii * 1.6, self.square_length / 2 - 0.5 - 4), 0.8, 4,
                                  color='lightgray', alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            self.ax.add_patch(
                patches.Rectangle((-self.square_length / 2 + 0.5, self.square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
                                  color='lightgray',
                                  alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            self.ax.add_patch(
                patches.Rectangle((self.square_length / 2 - 0.5 - 4, self.square_length / 2 - jj - 0.8 - ii * 1.6), 4,
                                  0.8,
                                  color='lightgray',
                                  alpha=0.5))
            ii += 1

    def _get_risk(self, vehs: List[dict], ):
        vehicles = Vehicles()
        i = 0
        for _veh in vehs:
            if 'a' not in _veh.keys(): _veh['a'] = 0
            if _veh['name'] == self.ego_veh_name:
                vehicles.set_ego(type='car', x0=_veh['x'], y0=_veh['y'], speed=_veh['v'],
                                 phi0=_veh['phi'] / 180 * np.pi, a0=_veh['a'])
            else:
                vehicles.add_obj(type='car', x0=_veh['x'], y0=_veh['y'], speed=_veh['v'],
                                 phi0=_veh['phi'] / 180 * np.pi, a0=_veh['a'])
                _veh['localID'] = i
                i += 1
        over_all_risk, vc, rc = vehicles.estimate_risk()
        if self.ana_mode:
            self.info_collect['delta_v'] = vehicles.obj[0].delta_v if len(vehicles.obj) > 0 else [0]
            self.info_collect['dis_weight'] = vehicles.obj[0].dis_de_curve if len(vehicles.obj) > 0 else [0]
            self.info_collect['risk_curve'] = vehicles.obj[0].risk_curve if len(vehicles.obj) > 0 else [0]
            self.info_collect['max_r_step'] = vehicles.obj[0].max_risk_step if len(vehicles.obj) > 0 else 0
            self.info_collect['time_weight'] = vehicles.obj[0].weight_t if len(vehicles.obj) > 0 else [0]
            self.info_collect['damage'] = vehicles.obj[0].damage if len(vehicles.obj) > 0 else [0]

        v_min, v_max = 100.000, -100.000
        for _veh in vehs:
            if _veh['name'] == self.ego_veh_name:
                _veh['risk'] = 'R=' + str(round(over_all_risk, 3))
            else:
                _veh['risk'] = round(vehicles.obj[_veh['localID']].risk, 3)
                v_min = min(round(vehicles.obj[_veh['localID']].risk, 3), v_min)
                v_max = max(round(vehicles.obj[_veh['localID']].risk, 3), v_max)
        return (v_min, v_max, over_all_risk, vc, rc)

    def draw(self, vehs: List[dict], ego_veh_name=None, t=None, add_text=None, step_norm=True, dis_range=50):
        vehs_in_plot_area = [_veh for _veh in vehs if self._is_in_plot_area(_veh)]  # list(dict)
        vehs_in_plot_area_name = [_veh['name'] for _veh in vehs_in_plot_area]

        if ego_veh_name == None:
            if len(vehs_in_plot_area_name) != 0 and self.ego_veh_name not in vehs_in_plot_area_name:
                self.ego_veh_name = random.choice(vehs_in_plot_area_name)
                self.risk_collect = {'time': [], 'risks': [], 'rc': [[], []], 'vc': [[], []]}
        else:
            self.ego_veh_name = ego_veh_name

        name_list = [m['name'] for m in vehs]
        if self.ego_veh_name in name_list:
            self.ego_veh = vehs[name_list.index(self.ego_veh_name)]
        elif add_text != None:
            add_text = 'name'

        vehs_in_risk_area = [_veh for _veh in vehs_in_plot_area if self._is_in_risk_area(_veh, dis_range)]  # list(dict)
        reserved_plot_veh = [_veh_name for _veh_name in self.last_step_vehs_rec.keys() if
                             _veh_name in [v['name'] for v in vehs_in_plot_area]]  # list(name)
        droped_plot_veh = [_veh_name for _veh_name in self.last_step_vehs_rec.keys() if
                           _veh_name not in [v['name'] for v in vehs_in_plot_area]]

        # tic = time.time()
        v_min, v_max, over_all_risk, vc, rc = self._get_risk(vehs_in_risk_area)
        # print('\r', '{:.1f} : {:.3f}'.format(float(t), (time.time() - tic) * 1000), end='')

        if not t == None:
            if self.time_handle == None:
                self.time_handle = self.ax.text(0.03, 0.82,
                                                'Sim_time={:.1f}s'.format(float(
                                                    t)) + '\nRisk:{:.3f} \nPred. Collision: {:d} \nReal Collision: {:d}'.format(
                                                    over_all_risk, vc, rc),
                                                transform=self.ax.transAxes, fontsize=18)
            else:
                self.time_handle.set(
                    text='Sim_time={:.1f}s'.format(float(t)) + '\nRisk:{:.3f} \nPred. Collision: {:d} \nReal Collision: {:d}'.format(over_all_risk, vc,rc))

        if step_norm:
            self.c_norm = colors.Normalize(vmin=v_min, vmax=v_max, clip=True)
        self._set_color(vehs_in_plot_area, vehs_in_risk_area)

        for _veh in vehs_in_plot_area:
            if not _veh['name'] in reserved_plot_veh:
                rec_handle = self._draw_rotate_rec(_veh)
                self.ax.add_patch(rec_handle)
                self.last_step_vehs_rec[_veh['name']] = rec_handle
                if add_text != None:
                    text_handle = self.ax.text(_veh['x'] + 1, _veh['y'] + 2, str(_veh[add_text]), fontsize=18)
                    if _veh['name'] == self.ego_veh_name: text_handle.set(color='red')
                    self.last_step_vehs_text[_veh['name']] = text_handle
            else:
                bottom_left_x, bottom_left_y, _ = rotate_coordination(-_veh['l'] / 2, _veh['w'] / 2, 0,
                                                                      -_veh['phi'])
                self.last_step_vehs_rec[_veh['name']].set(xy=(_veh['x'] + bottom_left_x, _veh['y'] + bottom_left_y),
                                                          angle=_veh['phi'] - 90, edgecolor=_veh['edge_color'],
                                                          facecolor=_veh['face_color'])
                if add_text != None:
                    self.last_step_vehs_text[_veh['name']].set(x=_veh['x'] + 1, y=_veh['y'] + 2,
                                                               text=str(_veh[add_text]))

        for _veh_name in droped_plot_veh:
            self.last_step_vehs_rec[_veh_name].remove()
            self.last_step_vehs_rec.pop(_veh_name)
            if add_text != None:
                self.last_step_vehs_text[_veh_name].remove()
                self.last_step_vehs_text.pop(_veh_name)

        self.risk_collect['time'].append(t)
        self.risk_collect['risks'].append(over_all_risk)
        if rc == 1:
            self.risk_collect['rc'][0].append(t)
            self.risk_collect['rc'][1].append(over_all_risk)
        elif vc == 1:
            self.risk_collect['vc'][0].append(t)
            self.risk_collect['vc'][1].append(over_all_risk)
        offset_rc = np.array(self.risk_collect['rc']).T
        offset_vc = np.array(self.risk_collect['vc']).T
        line_color = 'green' if vc != 1 else 'red'
        self.risk_line_handle.set(xdata=self.risk_collect['time'], ydata=self.risk_collect['risks'], color=line_color)
        self.coll_scatter_handle.set_offsets(offset_rc)
        self.pre_coll_scatter_handle.set_offsets(offset_vc)
        self.ax_line.set_xlim(min(self.risk_collect['time']), max(self.risk_collect['time']) + 5)
        # self.ax_line.set_ylim(0, max(self.risk_collect['risks']) + 0.4)
        self.ax_line.set_ylim(min(self.risk_collect['risks']), max(self.risk_collect['risks']) + 0.4)

        if self.ana_mode:
            self.dis_weight_handle.set(xdata=np.linspace(0, 3, 31), ydata=self.info_collect['dis_weight'])
            self.max_r_handle_1.set_offsets([self.info_collect['max_r_step'] / 10,
                                             self.info_collect['dis_weight'][self.info_collect['max_r_step']]])
            self.time_weight_handle.set(xdata=np.linspace(0, 3, 31), ydata=self.info_collect['time_weight'])
            self.max_r_handle_11.set_offsets([self.info_collect['max_r_step'] / 10,
                                             self.info_collect['time_weight'][self.info_collect['max_r_step']]])
            self.ax_ana_1.set_xlim(0, 3)
            self.ax_ana_1.set_ylim(0, 1)

            self.delta_v_handle.set(xdata=np.linspace(0, 3, 31), ydata=self.info_collect['delta_v'])
            self.max_r_handle_2.set_offsets(
                [self.info_collect['max_r_step'] / 10, self.info_collect['delta_v'][self.info_collect['max_r_step']]])
            self.ax_ana_2.set_xlim(0, 3)
            self.min_y = min(min(self.info_collect['delta_v']), self.min_y)
            self.max_y = max(max(self.info_collect['delta_v']), self.max_y)
            self.ax_ana_2.set_ylim(self.min_y, self.max_y)
            # self.ax_ana_2.set_ylim(-8, 8)

            self.risk_curve_handle.set(xdata=np.linspace(0, 3, 31), ydata=self.info_collect['risk_curve'])
            self.max_r_handle_3.set_offsets(
                [self.info_collect['max_r_step'] / 10,
                 self.info_collect['risk_curve'][self.info_collect['max_r_step']]])
            # self.damage_curve_handle.set(xdata=np.linspace(0, 3, 31), ydata=self.info_collect['damage'])
            # self.max_r_handle_3.set_offsets(
            #     [self.info_collect['max_r_step'] / 10,
            #      self.info_collect['damage'][self.info_collect['max_r_step']]])

            self.text_handle.set_text('$\omega_d:$' + '{:.3f}'.format(self.info_collect['dis_weight'][self.info_collect['max_r_step']]) +
                                      '$* \omega_t: $' + '{:.3f}'.format(self.info_collect['time_weight'][self.info_collect['max_r_step']]) +
                                      ' = ' + '{:.3f}'.format(self.info_collect['dis_weight'][self.info_collect['max_r_step']] *
                                                  self.info_collect['time_weight'][self.info_collect['max_r_step']]))
            self.ax_ana_3.set_xlim(0, 3)
            self.min_y_1 = min(min(self.info_collect['risk_curve']), self.min_y_1)
            self.max_y_1 = max(max(self.info_collect['risk_curve']), self.max_y_1)
            # self.min_y_1 = min(min(self.info_collect['risk_curve']), min(self.info_collect['damage']), self.min_y_1)
            # self.max_y_1 = max(max(self.info_collect['risk_curve']), max(self.info_collect['damage']), self.max_y_1)
            self.ax_ana_3.set_ylim(self.min_y_1, self.max_y_1)
            # self.ax_ana_3.set_ylim(-0.1, 0.6)

    def draw_for_gif(self, info: Tuple[List[dict], float], ego_veh_name=None, add_text=None, step_norm=True):
        self.draw(vehs=info[0], ego_veh_name=ego_veh_name, t=info[1], add_text=add_text, step_norm=step_norm)

    def _draw_rotate_rec(self, veh):
        x, y, a, l, w = veh['x'], veh['y'], veh['phi'], veh['l'], veh['w']
        bottom_left_x, bottom_left_y, _ = rotate_coordination(-veh['l'] / 2, veh['w'] / 2, 0, -veh['phi'])
        rec = patches.Rectangle((veh['x'] + bottom_left_x, veh['y'] + bottom_left_y), w, l, ec=veh['edge_color'],
                                fc=veh['face_color'], angle=-(90 - a), zorder=50)
        return rec

def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * pi / 180
    transformed_x = orig_x * cos(coordi_rotate_d_in_rad) + orig_y * sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * sin(coordi_rotate_d_in_rad) + orig_y * cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d

def draw_pics():
    with open(r'intersection_traffic_data.pkl', 'rb') as f:
        traffic_data = pickle.load(f)

    ego_veh_name = 'carflow_1.1.54'
    draw_times = np.arange(76, 105, 0.1)
    save_time = [77.9, 78.7, 79.9, 81.6, 82.7, 83.5, 85.9, 88.6, 93.8]

    render = Render(xyticklim=[-45,45,-45,45])  #
    print('start rendering')
    for time in draw_times:
        print('\r', '{:.1f} / {:.1f}'.format(float(time), np.max(draw_times)), end='')
        vehs = traffic_data[round(time, 1)]
        render.draw(vehs, ego_veh_name=ego_veh_name, t=time, add_text='risk')  # ego_veh_name=ego_veh_name,
        if round(time, 1) in save_time:
            plt.savefig(r'dyn_pics/' + str(round(time, 1)) + '.jpg')
        plt.pause(0.1)


if __name__ == '__main__':    
    draw_pics()