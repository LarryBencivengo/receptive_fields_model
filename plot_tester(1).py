
# coding: utf-8

# In[52]:

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
from typing import List, Any, Dict


from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, Semaphore
from functools import partial
import traceback
# import pytest

#import glia.config as config
#from .analysis import last_spike_time
#from .pipeline import get_unit
#from .functional import zip_dictionaries
#from .types import Unit
#from glia.config import logger

Seconds = float
ms = float
UnitSpikeTrains = List[Dict[str,np.ndarray]]


def axis_generator(ax,transpose=False):
    if isinstance(ax,matplotlib.axes.Axes):
        yield(ax)
    else:
        if transpose:
            axes = np.transpose(ax)
        else:
            axes = ax
        for handle in axes.reshape(-1):
            yield(handle)

def multiple_figures(nfigs, nplots, ncols=4, nrows=None, ax_xsize=4, ax_ysize=4, subplot_kw=None):
    figures = []
    axis_generators = []

    if not nrows:
        nrows = int(np.ceil(nplots/ncols))

    for i in range(nfigs):
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
        axis = axis_generator(ax)
        figures.append(fig)
        axis_generators.append(axis)

    return (figures, axis_generators)

def subplots(nplots, ncols=4, nrows=None, ax_xsize=4, ax_ysize=4, transpose=False,subplot_kw=None):
    if not nrows:
        nrows = int(np.ceil(nplots/ncols))

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
    axis = axis_generator(ax,transpose=transpose)

    return fig, axis

def plot_pdf_path(directory,name):
    return os.path.join(directory,name+".pdf")

def open_pdfs(plot_directory, unit_ids,unit_name_lookup=None):
    if unit_name_lookup is not None:
        return {unit_id: PdfPages(plot_pdf_path(plot_directory,unit_name_lookup[unit_id])) for unit_id in unit_ids}
    else:
        return {unit_id: PdfPages(plot_pdf_path(plot_directory, unit_id)) for unit_id in unit_ids}

def add_figure_to_unit_pdf(fig,unit_id,unit_pdfs):
    unit_pdfs[unit_id].savefig(fig)
    return (unit_id, fig)

def add_to_unit_pdfs(id_figure_tuple,unit_pdfs):
    print("Adding figures to PDFs")
    for unit_id,fig in tqdm(id_figure_tuple):
        add_figure_to_unit_pdf(fig,unit_id,unit_pdfs)

def close_pdfs(unit_pdfs):
    print("saving PDFs")
    for unit_id,pdf in tqdm(unit_pdfs.items()):
        pdf.close()

def close_figs(figures):
    for fig in figures:
        plt.close(fig)

def save_unit_figs(figures, filenames):
    for fig,name in zip(figures,filenames):
        fig.savefig(name)

def save_unit_fig(filename, unit_id, fig):
    logger.debug("Saving {} for {}".format(filename,unit_id))
    name = unit_id
    fig.savefig(os.path.join(config.plot_directory,name,filename+".png"))

def save_retina_fig(filename, fig):
    logger.debug("Saving {} for retina".format(filename))
    fig.savefig(os.path.join(config.plot_directory,"00-all",filename+".png"))

def isi_histogram(unit_spike_trains: UnitSpikeTrains, bin_width: Seconds=1/1000,
                  time: (Seconds, Seconds)=(0, 100/1000), average=True,
                  fig_size=(15, 30)) -> (Any):
    channels = unit_spike_trains.keys()

    channels = [np.diff(c) for c in channels]
    # Unit is seconds so x is in ms for x/1000
    bins = np.arange(time[0], time[1], bin_width)
    fig = plt.figure(figsize=fig_size)

    if average:
        # flatten array
        all_isi = np.hstack([c for c in channels if c is not None])

        ax = fig.add_subplot(111)
        ax.hist(all_isi, bins)
    else:
        subp = subplot_generator(channels,5)
        for channel in channels:
            ax = fig.add_subplot(*next(subp))
            ax.hist(channel, bins)


def draw_spikes(ax, spike_train, ymin=0,ymax=1,color="black",alpha=0.5):
    "Draw each spike as black line."
    # draw_spike = np.vectorize(lambda s: ax.vlines(s, ymin, ymax,colors=color,alpha=alpha))
    # for spike in spike_train:
    #     draw_spike(spike)
    ax.vlines(spike_train, ymin, ymax,colors=color,alpha=alpha)


# Helpers

def subplot_generator(n_charts, num_cols):
    """Generate arguments for matplotlib add_subplot.

    Must use * to unpack the returned tuple. For example,

    >> fig = plt.figure()
    <matplotlib.figure.Figure at 0x10fdec7b8>
    >> subp = subplot_generator(4,2)
    >> fig.add_subplot(*next(subp))
    <matplotlib.axes._subplots.AxesSubplot at 0x112cee6d8>
    (NOTE doctest not running)
    """

    if type(n_charts) is list:
        n_charts = len(n_charts)

    num_rows = n_charts // num_cols + (n_charts % num_cols != 0)
    n = 1
    while n <= n_charts:
        yield (num_rows, num_cols, n)
        n += 1

def create_polar_scatterplot(stimulus_analytics: dict, ax = None):
    if ax is None:
        fig =plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig = None
    angles = {k:count_items(v) for k,v in stimulus_analytics.items()}
    theta = []
    r = []
    area = []
    for a, counts in angles.items():
        for spike_count, number_of_occurrences  in counts.items():
            theta.append(a)
            r.append(spike_count)
            area.append(number_of_occurrences^3)
    c = ax.scatter(theta, r, area)
    c.set_alpha(0.75)
    if fig is not None:
        return fig


def count_items(my_list):
    to_return={}
    for i in my_list:
        try:
            to_return[i]+=1
        except:
            to_return[i]=1
    return to_return

def plot_ifr(ax_gen, unit, ylim=None, legend=False):
    color=iter(plt.cm.rainbow(np.linspace(0,1,20))) #this is a hack, should not be 20

    for stimulus, ifr_list in unit.items():
        c = next(color)
        stim = eval(stimulus)
        l = "speed:"+ str(stim["speed"]) + ", width:" + str(stim["width"]) + ", bar_color:"+str(stim["barColor"])
        for trial in ifr_list:
            ax = next(ax_gen)
            ax.plot(trial, color=c)
            ax.set_title(l)
            if ylim is not None:
                ax.set_ylim(ylim)

def plot_direction_selectively(ax, unit_id, bar_firing_rate, bar_dsi, legend=False):
    """Plot the average for each speed and width."""
    # we will accumulate by angle in this dictionary and then divide
    analytics = by_speed_width_then_angle(bar_firing_rate[unit_id])
    speed_widths = analytics.keys()
    speeds = sorted(list(set([speed for speed,width in speed_widths])))
    widths = sorted(list(set([width for speed,width in speed_widths])))
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(speeds))))
    w = iter(np.linspace(1,5,len(widths)))
    speed_style = {speed: next(color) for speed in speeds}
    width_style = {width: next(w) for width in widths}

    for speed_width, angle_dictionary in analytics.items():
        speed, width = speed_width
        line_angle = []
        line_radius = []
        for angle, average_number_spikes in angle_dictionary.items():
            line_angle.append(angle)
            line_radius.append(average_number_spikes)
        # connect the line
        line_angle, line_radius = sort_two_arrays_by_first(line_angle,line_radius)
        line_angle.append(line_angle[0])
        line_radius.append(line_radius[0])
        ax.plot(line_angle,line_radius, linewidth=width_style[width], color=speed_style[speed], label=speed_width)

    ax.set_title('Unit: '+str(unit_id))
    ax.set_ylabel("Firing rate (Hz)")
    speed_style["overall"] = "white"
    unique_speed_width = sorted(speed_widths, key=f_get_key(1))
    unique_speed_width = sorted(unique_speed_width, key=f_get_key(0))

    columns = unique_speed_width + ["overall"]
    colors = [speed_style[speed] for speed,width in unique_speed_width] + ['white']
    cells = [["{:1.3f}".format(bar_dsi[unit_id][speed_width]) for speed_width in columns],              ["{:1.3f}".format(bar_osi[unit_id][speed_width]) for speed_width in columns]]

    table = ax.table(cellText=cells,
                      rowLabels=['DSI',"OSI"],
                      colLabels=columns,
                     colColours=colors,
                        loc='bottom', bbox = [0,-0.2,1,0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    if legend is True:
        ax.legend()

def plot_units(unit_plot_function, c_unit_fig, *units_data, nplots=1, ncols=1,
               nrows=None, ax_xsize=2, ax_ysize=2, figure_title=None, transpose=False,
               subplot_kw=None):
    """Create a giant figure with one or more plots per unit.

    c_unit_fig determines what happens to fig when produced
    transpose==True will yield axes by column

    Must supply an even number of arguments that alternate function, units. If one pair is provided,
    ncols will determine the number of columns. Otherwise, each unit will get one row."""
    logger.info("plotting units")
    number_of_units = len(units_data[0].keys())

    processes = config.processes

    all_data = zip_dictionaries(*units_data)

    def data_generator():
        for unit_id, data in all_data:
            yield (unit_id, data)
                # ax_xsize, ax_ysize, figure_title, subplot_kw, semaphore)

    pool = Pool(processes)
    logger.info("passing tasks to pool")
    plot_worker = partial(_plot_worker, unit_plot_function, c_unit_fig, nplots,
                ncols, nrows, ax_xsize, ax_ysize, figure_title, transpose, subplot_kw)
    list(pool.imap_unordered(plot_worker, tqdm(data_generator(), total=number_of_units)))
    pool.close()
    pool.join()


def _plot_worker(plot_function, c_unit_fig, nplots, ncols, nrows, ax_xsize,
        ax_ysize, figure_title, transpose, subplot_kw, args):
    "Use with functools.partial() so function only takes args."

    logger.debug("plot worker")

    unit_id, data = args
        # ax_ysize, figure_title, subplot_kw, semaphore) = args
    if len(data)==1:
        data = data[0]
    fig = plot(plot_function, data, nplots, ncols=ncols, nrows=nrows,
        ax_xsize=ax_xsize, ax_ysize=ax_ysize,
        figure_title=figure_title, transpose=transpose, subplot_kw=subplot_kw)

    logger.debug("Plot worker successful for {}".format(unit_id))
    c_unit_fig(unit_id,fig)
    plt.close(fig)


def plot_each_by_unit(unit_plot_function, units, ax_xsize=2, ax_ysize=2,
               subplot_kw=None):
    "Iterate each value by unit and pass to the plot function."
    # number of units
    # number of values
    number_of_plots = len(get_unit(units)[1].values())
    number_of_units = len(list(units.keys()))

    # if nrows*ncols > 100:
    #     nrows = int(np.floor(100/ncols))
    #     print("only plotting first {} units".format(nrows))

    # fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
    # axis = axis_generator(ax)

    multiple_figures(number_of_units, number_of_plots)
    i = 0
    for unit_id, value in units.items():
        if i>=100:
            break
        else:
            i+=1

        if type(value) is dict:
            gen = value.items()
        else:
            gen = value
        for v in gen:
            cur_ax = next(axis)
            unit_plot_function(cur_ax,unit_id,v)
    return fig


def plot_from_generator(plot_function, data_generator, nplots, ncols=4, ax_xsize=7, ax_ysize=10, ylim=None, xlim=None, subplot_kw=None):
    "plot each data in list_of_data using plot_function(ax, data)."
    nrows = int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
    axis = axis_generator(ax)

    for data in data_generator():
        cur_ax = next(axis)
        plot_function(cur_ax, data)
        if ylim is not None:
            cur_ax.set_ylim(ylim)
        if xlim is not None:
            cur_ax.set_xlim(xlim)

    return fig


def plot(plot_function, data, nplots=1, ncols=1, nrows=None, ax_xsize=4,
            ax_ysize=4, figure_title=None, transpose=False, subplot_kw=None):
    fig, axes = subplots(nplots, ncols=ncols, nrows=nrows, ax_xsize=ax_xsize, ax_ysize=ax_ysize,
                        transpose=transpose, subplot_kw=subplot_kw)
    try:
        plot_function(fig, axes, data)
    except Exception as e:
        print("Plot function ({}) failed: ".format(plot_function),e)
        traceback.print_tb(e.__traceback__)
    if figure_title is not None:
        fig.suptitle(figure_title)
    return fig

def plot_each_for_unit(unit_plot_function, unit, subplot_kw=None):
    "Single unit version of plot_each_by_unit."
    unit_id = unit[0]
    value = unit[1]
    ncols = len(value.values())
    fig, ax = plt.subplots(1, ncols, subplot_kw=subplot_kw)
    axis = axis_generator(ax)
#     axis = iter([ax])


    if type(value) is dict:
        gen = value.items()
    else:
        gen = value
    for v in gen:
        cur_ax = next(axis)
        unit_plot_function(cur_ax,unit_id,v)
    return fig

def c_plot_solid(ax):
    ax.set_title("Unit spike train per SOLID")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials")

def _axis_continuation_helper(continuation,axis):
    continuation(axis)
    return axis

def axis_continuation(function):
    """Takes a lambda that modifies the axis object, and returns the axis.

    This enables a pipeline of functions that modify the axis object"""

    return partial(_axis_continuation_helper,function)

def plot_spike_trains(fig, axis_gen, data,prepend_start_time=0,append_lifespan=0,
                      continuation=c_plot_solid, ymap=None):
    ax = next(axis_gen)
    for i,v in enumerate(data):
        stimulus, spike_train = (v["stimulus"], v["spikes"])

        # use ymap to keep charts aligned if a comparable stimulus was not run
        # i.e., if each row is a different bar width but one chart is sparsely sampled
        if ymap:
            trial = ymap(stimulus)
        else:
            trial = i

        lifespan = stimulus['lifespan']
        '''
        logger.debug("plot_spike_trains ({}) iteration: {}, lifespan: {}".format(
            stimulus["stimulusType"],trial,lifespan))
        if lifespan > 120:
            logger.debug("skipping stimulus longer than 120 seconds")
        '''

        if spike_train.size>0:
            draw_spikes(ax, spike_train, ymin=trial+0.3,ymax=trial+1)

        stimulus_end = prepend_start_time + lifespan
        duration = stimulus_end + append_lifespan
        if stimulus_end!=duration:
            # this is for solid
            ax.fill([0,prepend_start_time,prepend_start_time,0],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)
            ax.fill([stimulus_end,duration,duration,stimulus_end],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)
        else:
            # draw all gray for all others
            ax.fill([0,lifespan,lifespan,0],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)


    continuation(ax)

# @pytest.fixture(scope="module")
# def channels():
#     import files
#     return read_mcs_dat('tests/sample_dat/')


def group_lifespan(group):
    return sum(list(map(lambda x: x["stimulus"]["lifespan"], group)))


def raster_group(fig, axis_gen, data):
    """Plot all spike trains for group on same row.

    Expects data to be a list of lists """
    ax = next(axis_gen)
    trial = 0
    longest_group = max(map(group_lifespan, data))

    for group in data:
        # x offset for row, ie relative start_time
        offset = 0
        end_time = 0
        for i,v in enumerate(group):
            stimulus, spike_train = (v["stimulus"], v["spikes"])
            lifespan = stimulus['lifespan']
            end_time += lifespan


            if lifespan > 60:
                logger.warning("skipping stimulus longer than 60 seconds")
                continue
            if spike_train.size>0:
                draw_spikes(ax, spike_train+offset, ymin=trial+0.3,
                    ymax=trial+1)

            kwargs = {"facecolor": stimulus["backgroundColor"],
                      "edgecolor": "none",
                      "alpha": 0.1}
            if stimulus['stimulusType']=='BAR':
                kwargs['hatch'] = '/'
            ax.fill([offset,end_time,end_time,offset],
                    [trial,trial,trial+1,trial+1],
                    **kwargs)
            offset = end_time

        if offset<longest_group:
            ax.fill([offset,longest_group,longest_group,offset],
                    [trial,trial,trial+1,trial+1],
                    facecolor="black",
                    edgecolor="none", alpha=1)

        trial += 1

    ax.set_title("Unit spike train per group")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials")
    ax.set_xlim((0,longest_group))
    ax.set_ylim((0,trial))



# In[146]:

import math
import random

global tick_rate, ticks, last_tick   # ticks per second, current # of ticks, arbitrarily large number (3+ hours)
tick_rate, ticks, last_tick = 1000, 0, 11000000

# New Model Neuron for Receptive Field Mapping

class Neuron(object):
    '''
    This model neuron will generate pseudo-random spike trains in response to a simulated stimulus.
    The model's firing rate will vary in response to the stimulus as a result of several neuron parameters and
    the location of the stimulus relative to the neuron at any given moment in time.

    '''
    # Class Variables
    id = 0

    def __init__(self, channel, retina, neuron_type = None, x = 0, y = 0):
        self.id = Neuron.id
        Neuron.id += 1
        self.neuron_type = neuron_type
        self.channel = channel
        self.retina = retina
        self.x, self.y = x, y        # (x, y) offset from channel in microns. Default is (0,0). NOT YET CODED!
        self.spike_train = []

    def spiking(self, firing_rate = 1):
        '''
        spiking is a function that stochastically generates spike trains for a neuron.  Default firing_rate
        is 1 Hz.
        '''
        if random.random() <= firing_rate / tick_rate:
            self.spike_train.append(ticks / tick_rate)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def get_spike_train(self):
        return self.spike_train[:]

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return 'Neuron #' + str(self.id)

class RGC(Neuron):
    '''
    Retinal Ganglion Cells are the primary Neuron-types whose electrical impulses are detected by the
    channels on an mea.  The subtypes represented by this class are primarily theoretical in nature,
    including different properties tested by the model in question.  They do not necessarily correspond
    to actual RGC subtypes found in vivo.
    '''
    # subtypes is a dictionary whose keys are the various subtypes of RGCs and whose values are a dictionary of
    # the corresponding attributes of each subtype, such as radius [of receptive field], latency of ON response,
    # spontaneous_rate, ON_rate [Light-Dependent], etc.
    subtypes = {
        'standard' :
        {
        'soma' : 20,             # diameter of cell body in microns
        'radius' : 0,            # radius of receptive field in microns
        'eccentricity' : 0,      # eccentricity of receptive field
        'sensitivity' : 0,       # threshold of light intensity below which response is attenuated
        'orientation' : None,    # orientation selectivity
        'direction' : None,      # directional selectivity
        'wavelength' : None,     # wavelength selectivity
        'spontaneous_rate' : 1.0,# (Hz)
        'ON_OFF' : 'ON',         # light response: 'ON', 'OFF' or 'ON/OFF'
        'latency_ON' : 0,        # delay between onset of stimulus and change in firing rate
        'accel_ON' : 0.01,       # larger values cause rate to change more slowly
        'trans_ON' : False,      # indicates whether cell is transient ON or sustained ON
        'ON_rate' : 10.0,        # (Hz) ON firing rate for cells with ON response
        'ON_duration' : 50.0,    # (msec) duration of full ON_rate of firing
        'ON_decay' : 0,          # any positive value indicates the rate at which the ON_rate decays over time
        'latency_OFF' : 0,       # delay between stimulus off and change in firing rate
        'accel_OFF' : 0.01,      # larger values cause rate to change more slowly
        'OFF_rate' : 0.0,        # (Hz) OFF firing rate for cells with OFF response (should differ from ON_rate)
        'OFF_duration' : 50.0,   # (msec) duration of full OFF_rate of firing
        'OFF_decay' : 0.0        # any positive value indicates the rate at which the OFF_rate decays over time
        },
        'type1' :
        {
        'soma' : 20, 'radius' : 0, 'eccentricity' : 0, 'sensitivity' : 0, 'orientation' : None, 'direction' : None,
        'wavelength' : None, 'spontaneous_rate' : 1.0, 'ON_OFF' : 'ON', 'latency_ON' : 100, 'accel_ON' : 0.01,
        'trans_ON' : False, 'ON_rate' : 10.0, 'ON_duration' : 500.0, 'ON_decay' : 1000, 'latency_OFF' : 50,
        'accel_OFF' : 0.01, 'OFF_rate' : 0.0, 'OFF_duration' : 500.0, 'OFF_decay' : 0.0
        },
        'type2' :
        {
        'soma' : 20, 'radius' : 200, 'eccentricity' : 0, 'sensitivity' : 0, 'orientation' : None, 'direction' : None,
        'wavelength' : None, 'spontaneous_rate' : 1.0, 'ON_OFF' : 'ON', 'latency_ON' : 100, 'accel_ON' : 0.01,
        'trans_ON' : False, 'ON_rate' : 10.0, 'ON_duration' : 500.0, 'ON_decay' : 1000, 'latency_OFF' : 50,
        'accel_OFF' : 0.01, 'OFF_rate' : 0.0, 'OFF_duration' : 500.0, 'OFF_decay' : 0.0
        }
    }

    def __init__(self, channel, retina, subtype = 'standard'):
        Neuron.__init__(self, channel, retina, neuron_type = 'RGC')
        if not subtype:
            subtype = 'standard'
        # constant attributes of a RGC
        self.subtype = subtype
        self.soma = RGC.subtypes[self.subtype]['soma']
        self.radius = RGC.subtypes[self.subtype]['radius']
        self.eccentricity = RGC.subtypes[self.subtype]['eccentricity']
        self.sensitivity = RGC.subtypes[self.subtype]['sensitivity']
        self.orientation = RGC.subtypes[self.subtype]['orientation']
        self.direction = RGC.subtypes[self.subtype]['direction']
        self.wavelength = RGC.subtypes[self.subtype]['wavelength']
        self.spontaneous_rate = RGC.subtypes[self.subtype]['spontaneous_rate']
        self.ON_OFF = RGC.subtypes[self.subtype]['ON_OFF']
        self.latency_ON = RGC.subtypes[self.subtype]['latency_ON']
        self.accel_ON = RGC.subtypes[self.subtype]['accel_ON']
        self.trans_ON = RGC.subtypes[self.subtype]['trans_ON']
        self.ON_rate = RGC.subtypes[self.subtype]['ON_rate']
        self.ON_duration = RGC.subtypes[self.subtype]['ON_duration']
        self.ON_decay = RGC.subtypes[self.subtype]['ON_decay']
        self.latency_OFF = RGC.subtypes[self.subtype]['latency_OFF']
        self.accel_OFF = RGC.subtypes[self.subtype]['accel_OFF']
        self.OFF_rate = RGC.subtypes[self.subtype]['OFF_rate']
        self.OFF_duration = RGC.subtypes[self.subtype]['OFF_duration']
        self.OFF_decay = RGC.subtypes[self.subtype]['OFF_decay']

        # variable attributes of a RGC
        self.firing_rate = self.spontaneous_rate
        self.target_rate = self.firing_rate
        self.time_ON = last_tick     # last tick set to arbitrarily high value (3+ hours at tick_rate = 1000)
        self.time_OFF = last_tick    # last tick set to arbitrarily high value (3+ hours at tick_rate = 1000)
        self.distance = last_tick    # distance to stimulus set to arbitratily high value
        self.light_ON = False
        self.intensity = 0
        self.color = 'white'

    def spiking(self, new_firing_rate = None):
        '''
        spiking is a function that stochastically generates spike trains for a neuron based on firing_rate.
        Default firing_rate is the neuron's spontaneous firing rate.

        return:   None
        '''
        if new_firing_rate:
            self.firing_rate = new_firing_rate
        if random.random() <= self.firing_rate / tick_rate:
            self.spike_train.append(ticks / tick_rate)

    def stimulus_response(self, light_ON, intensity, distance, color = 'white'):
        '''
        This function alters the neuron's firing rate in response to a change in the current state of the stimulus.
        It is called every time the stimulus updates itself so that neurons can update their response.

        light_ON:    boolean: True when the light is ON, False otherwise
        intensity:   int: an integer value indicating the relative brightness of the stimulus
        distance:    float: the distance in microns between the nearest edge of a stimulus and the center of the soma

        return:   None
        '''

        # ************ Alter target firing rate in response to a change in the state of the stimulus ***************
        #
        # In general, when the target_rate of a cell does not match the firing_rate appropriate for the stimulus,
        # this indicates that the cell has detected a *change* in the stimulus which will provoke a change in the
        # cell's internal state. latency-ON and latency_OFF introduce a delay between the change in the stimulus and
        # the cell's response which is kept track of by setting time_ON or time_OFF as a signal for when to change
        # self.firing_rate.
        #
        #

        # adjust variable parameters according to input from stimulus; needed for update_neuron() calls
        self.light_ON = light_ON
        self.intensity = intensity
        self.distance = distance
        self.color = color

        # when the light is ON for an ON cell or an ON/OFF cell
        if light_ON and (self.ON_OFF == 'ON' or self.ON_OFF == 'ON/OFF'):

            # stimulus within receptive field of neuron
            if distance <= 0.5 * self.soma + self.radius:
                if self.target_rate != self.ON_rate:   # this is a new response due to a change in the stimulus
                    self.target_rate = self.ON_rate
                    self.time_ON = ticks + self.latency_ON * tick_rate / 1000  # set time to increase firing_rate
                    self.time_OFF = self.time_ON + self.ON_duration * tick_rate / 1000 # set time for decay to begin

            # stimulus outside the receptive field of neuron
            else:
                if self.ON_OFF == 'ON':
                    if self.target_rate != self.spontaneous_rate:  # new response due to a change in the stimulus
                        self.target_rate = self.spontaneous_rate
                        self.time_OFF = ticks + self.latency_OFF * tick_rate / 1000  # set time to decrease firing_rate
                        self.time_ON = last_tick

                #elif self.ON_OFF == 'ON/OFF' and ticks < self.time_ON:
                    #if self.target_rate != self.spontaneous_rate:  # new response due to a change in the stimulus
                        #self.target_rate = self.spontaneous_rate
                        #self.time_ON = self.time_OFF + self.latency_ON * tick_rate / 1000 # time for OFF response to begin

        # when light is ON for an OFF cell
        elif light_ON and self.ON_OFF == 'OFF':
            if self.target_rate != self.ON_rate:    # OFF cell will now be 'primed' to switch ON when light turns OFF
                self.target_rate = self.ON_rate
                self.time_ON = last_tick
                self.time_OFF = last_tick


        # when the light is OFF for an ON cell
        elif not light_ON and self.ON_OFF == 'ON':
            if self.target_rate != self.spontaneous_rate:  # new response due to a change in the stimulus
                self.target_rate = self.spontaneous_rate
                self.time_OFF = ticks + self.latency_OFF * tick_rate / 1000  # set time to decrease firing_rate
                self.time_ON = last_tick

        # when the light is OFF for an OFF cell
        elif not light_ON and self.ON_OFF == 'OFF':
            if self.target_rate != self.OFF_rate:     # new response due to a change in the stimulus
                self.target_rate = self.OFF_rate
                self.time_ON = ticks + self.latency_ON * tick_rate / 1000 # time for OFF response to begin
                self.time_OFF = self.time_ON + self.OFF_duration * tick_rate / 1000 # time for OFF

        # when the light is OFF for an ON/OFF cell
        elif not light_ON and self.ON_OFF == 'ON/OFF':
            pass       # not yet coded!

    def update_neuron(self, new_firing_rate = None):
        '''
        This function is called every tick. The neuron updates its firing_rate and other variable attributes
        according to the current state of the stimulus and the neuron's inherent properties. It then calls
        spiking() to update the spike_train.

        new_firing_rate:   float: allows model to override the neuron's inherent firing rate with a user-specified rate

        return:   None
        '''
        # Set firing rate to user-selected value, ...
        if new_firing_rate:
            firing_rate = new_firing_rate

        # ... otherwise, determine whether firing rate needs to be altered due to a change in the
        # cell's *internal* state. Examples include decay in ON firing rate, changes in firing rate triggered
        # by a change in the stimulus but delayed by latency, attennuation of firing rate due to sub-threshold
        # light intensity or orientation/directional/wavelength selectivity, etc.
        #

        else:
            if self.light_ON and (self.ON_OFF == 'ON' or self.ON_OFF == 'ON/OFF'):

                # stimulus within receptive field of neuron
                if self.distance <= 0.5 * self.soma + self.radius:

                    # an ON_decay rate greater than zero means that firing_rate begins to decay after ON_duration
                    if ticks >= self.time_OFF and self.target_rate == self.ON_rate and self.ON_decay > 0.0:
                        self.firing_rate -= (self.firing_rate - self.spontaneous_rate) / (self.ON_decay * tick_rate / 1000)
                        if self.firing_rate < self.spontaneous_rate:
                            self.firing_rate = self.spontaneous_rate

                    # ramp up firing_rate toward the target (ON-rate)
                    elif ticks >= self.time_ON and ticks <= self.time_OFF and self.firing_rate != self.ON_rate:
                        self.firing_rate += (self.ON_rate - self.firing_rate) / (self.accel_ON * tick_rate / 1000)
                        if self.firing_rate > self.ON_rate:
                            self.firing_rate = self.ON_rate

                    # adjust firing rate for low light intensity
                    if self.intensity < self.sensitivity and self.firing_rate > self.ON_rate * (intensity / self.sensitivity):
                        self.firing_rate = self.firing_rate * (intensity / self.sensitivity)
                        if self.firing_rate < self.spontaneous_rate:
                            self.firing_rate = self.spontaneous_rate

            # light is OFF or stimulus outside the receptive field of neuron
            elif not self.light_ON or self.distance > 0.5 * self.soma + self.radius:

                # for an ON cell
                if self.ON_OFF == 'ON':

                    # an ON_decay rate greater than zero means that firing_rate begins to decay after ON_duration
                    if ticks < self.time_OFF:       # ON_rate still decaying
                        if self.firing_rate > self.spontaneous_rate and self.ON_decay > 0.0:
                            self.firing_rate -= (self.firing_rate - self.spontaneous_rate) / (self.ON_decay * tick_rate / 1000)
                            if self.firing_rate < self.spontaneous_rate:
                                self.firing_rate = self.spontaneous_rate

                    elif ticks >= self.time_OFF: # cell is switching OFF
                        if self.firing_rate > self.spontaneous_rate:
                            self.firing_rate -= (self.firing_rate - self.spontaneous_rate) / (self.accel_OFF * tick_rate / 1000)
                            if self.firing_rate < self.spontaneous_rate:
                                self.firing_rate = self.spontaneous_rate


        self.spiking(self.firing_rate)     # call spiking to generate spike_train



class Retina(object):
    '''
    A Retina instance keeps a dictionary of all Neurons in the Retina.
    '''
    id = 0

    def __init__(self, retina_type, experiment = None):
        '''
        experiment includes the variable <random_seed> so that stochastic data can be repeated, if necessary.
        '''
        self.id = Retina.id
        Retina.id += 1

        self.retina_type = retina_type
        self.experiment = experiment  # a dictionary with experimental parameters
        self.neuron_num = 0
        self.neurons = {}

    def add_neuron(self, channel, neuron_type = 'RGC', subtype = 'standard'):
        '''
        adds a neuron of the specified type and subtype (default is 'RGC' and 'standard')
        at the specified channel to self.neurons

        return:    None
        '''
        if neuron_type == None or neuron_type == 'Neuron':
            self.neurons[self.neuron_num] = Neuron(channel, self)
        elif neuron_type == 'RGC':
            self.neurons[self.neuron_num] = RGC(channel, self, subtype)
        else:
            raise ValueError('Incorrect (or No) neuron type specified')
        self.neuron_num += 1


class Stimulus(object):
    '''
    Updates stimulus, generates YAML file **(currently pseudo_YAML)**, and informs model neurons about changes
    in the stimulus.
    '''

    def __init__(self, stimulus, channels, mea):
        # read data from the stimulus dictionary
        self.stimulus_type, self.start_time, self.num_trials, self.num_angles, self.angles, self.num_widths, self.widths,         self.lifespans, self.num_speeds, self.speeds, self.num_intensities, self.intensities,         self.num_colors, self.wavelengths, self.delay,         self.image_width, self.image_height, self.pixel_size, self.pinhole = read_stimulus_dictionary(stimulus)

        # randomize the order of presentation of stimuli for the first trials
        # Not yet coded

        # read data from the mea dictionary
        self.channel_map, self.channel_spacing, self.offset_angle, self.offset_x, self.offset_y         = read_mea_dictionary(mea)

        if not channels:
            channels = [c for c in range(1, len(channel_map) + 1)]

        # determine the onset of the first stimulus and the number of ticks between updates for moving stimuli
        self.stim_start_tick = self.start_time * tick_rate    # calculate tick for first stim_start
        self.update_interval = tick_rate // self.speeds[0]

        # initialize the parameters that control the stimulus
        self.light_ON = False
        self.running = False
        self.updated = False
        self.last_update = last_tick
        self.next_update = self.stim_start_tick
        self.current_stimulus = 0
        self.end_experiment = False


        # generate YAML file
        # currently, a pseudo_YAML file is created which is a list of lists, one list
        # for each stimulus presented (only the Moving_Bar Stimulus exists at this point).  Data includes:
        # [angle, width, speed, intensity, color, lifespan(sec), stim_start_time (sec), stim_end_time(sec)]

    def update_stimulus(self):
        # updates the stimulus and passes information to the neurons

        # update stimulus

        # append to YAML file

        # pass information to neurons

        pass

    def stimulate_neurons(self, retina):
        '''
        Loops through all the neurons in retina, updating them on the current status of the stimulus.

        retina:   obj instance: the retina object of the current experiment

        return:   None
        '''

        pass



class Moving_Bar(Stimulus):
    '''
    Tracks and updates the position of the moving bar stimulus relative to each model neuron.
    '''
    def __init__(self, stimulus, channels, mea):
        # initialize a Moving_Bar Stimulus object
        Stimulus.__init__(self, stimulus, channels, mea)


        # get DATh_dict.  This dictionary contains the magnitudes of the vectors pointing from each channel
        # to the leading edge of a moving bar stimulus with angle theta at time t = 0.
        # These values are used to determine the location of the moving bar relative to the neuron at each tick.
        self.DATh_dict = create_DATh_dictionary(stimulus, channels, mea)

        # generate pseudo_YAML, the list that controls the stimuli (this is NOT a YAML file - but it should be)
        pseudo_YAML = []
        for angle in self.angles:
            for width in self.widths:
                for speed in self.speeds:
                    for intensity in self.intensities:
                        for color in self.wavelengths:
                            lifespan = (203 + width * self.pixel_size) / speed # 203 derived from actual Eye Candy YAMLs
                            pseudo_YAML.append([angle, width, speed, intensity, color, lifespan])
        self.pseudo_YAML = pseudo_YAML
        self.num_stimuli = len(self.pseudo_YAML)
        self.angle = self.pseudo_YAML[self.current_stimulus][0]
        self.width = self.pseudo_YAML[self.current_stimulus][1]
        self.speed = self.pseudo_YAML[self.current_stimulus][2]
        self.intensity = self.pseudo_YAML[self.current_stimulus][3]
        self.color = self.pseudo_YAML[self.current_stimulus][4]
        self.lifespan = self.pseudo_YAML[self.current_stimulus][5]


    def update_stimulus(self, retina):
        '''
        Updates the stimulus, then calls stimulus_response() function for each neuron with self.update_neurons()

        return:   boolean: False, unless all stimuli plus a final delay are complete, then True
        '''

        if ticks >= self.next_update:    # time for next stimulus update

            # time to start next stimulus
            if ticks >= self.stim_start_tick and not self.running:
                print('starting stimulus')
                self.running = True

                # update stimulus parameters
                self.light_ON = True
                self.angle = self.pseudo_YAML[self.current_stimulus][0]
                self.width = self.pseudo_YAML[self.current_stimulus][1]
                self.speed = self.pseudo_YAML[self.current_stimulus][2]
                self.intensity = self.pseudo_YAML[self.current_stimulus][3]
                self.color = self.pseudo_YAML[self.current_stimulus][4]
                self.lifespan = self.pseudo_YAML[self.current_stimulus][5]

                # update other variables
                self.last_update = ticks
                self.next_update = ticks + self.update_interval
                self.stim_start_time = self.stim_start_tick / tick_rate
                self.stim_end_tick = self.stim_start_tick + self.pseudo_YAML[self.current_stimulus][5] * tick_rate
                self.stim_end_time = self.stim_end_tick / tick_rate

                # update pseudo_YAML file
                self.pseudo_YAML[self.current_stimulus].append(self.stim_start_time)
                self.pseudo_YAML[self.current_stimulus].append(self.stim_end_time)

                # update neurons
                self.stimulate_neurons(retina)

            # time to end current stimulus
            elif self.running and ticks >= self.stim_end_tick:
                print('ending stimulus')
                self.running = False
                self.light_ON = False
                self.last_update = ticks
                self.next_update = ticks + self.delay * tick_rate
                self.stim_start_tick = self.next_update

                # update neurons
                self.stimulate_neurons(retina)

                # update num_stimulus
                self.current_stimulus += 1

                # check for end of experiment
                if self.current_stimulus == self.num_stimuli:
                    self.current_stimulus -= 1
                    self.end_experiment = ticks + self.delay * tick_rate
                    self.stim_start_tick = last_tick

            # time to update current stimulus
            else:
                self.last_update = ticks
                self.next_update = ticks + self.update_interval

                # update neurons
                self.stimulate_neurons(retina)

        # signal end of experiment, if all stimuli plus final delay are completed
        if self.end_experiment and ticks >= self.end_experiment:
            return True
        # else, continue
        else:
            return False


    def stimulate_neurons(self, retina):
        '''
        Loops through all the neurons in retina, updating them on the current status of the stimulus.

        retina:   obj instance: the retina object of the current experiment

        return:   None
        '''

        # loop through all neurons
        for unit in range(len(retina.neurons)):

            # find the distance between the neuron and the stimulus
            distance = self.calculate_distance(retina.neurons[unit])

            # stimulate the neuron to respond to the changed stimulus
            retina.neurons[unit].stimulus_response(self.light_ON, self.intensity, distance, self.color)


    def calculate_distance(self, unit):
        """
        Calculates the shortest distance between a given unit and a moving bar stimulus.  Accounts for whether
        the bar has reached or passed the unit.

        return:   float: the shortest distance between the bar stimulus and the unit
        """

        angle = self.pseudo_YAML[self.current_stimulus][0]
        width = self.pseudo_YAML[self.current_stimulus][1]
        lifespan = self.pseudo_YAML[self.current_stimulus][5]

        # find starting distance from leading edge of bar to unit at time t = 0
        DATh = self.DATh_dict[unit.channel][angle]      # does not currently account for any (x, y) offset from channel

        # use DATh, bar width and lifespan to calculate bar_speed, the actual rate of bar movement (microns/sec)
        bar_speed = (2 * DATh + width * self.pixel_size) / lifespan

        distance1 = DATh - (bar_speed * ticks / tick_rate)
        distance2 = distance1 + width * self.pixel_size

        return min(distance1, distance1)



class Experiment(object):
    '''
    Includes User-Interface, directs stimulus updates, prompts neurons to update and spike,
    records simulation results, etc.
    '''
    id = 0

    def __init__(self, retina = None, retina_type = None, treatment = None, stimulus_type = 'moving_bar'):
        self.stimulus_type = stimulus_type
        self.id = Experiment.id
        Experiment.id += 1

        # record information provided about the experiment to be simulated
        self.experiment = {'retina' : retina, 'retina_type' : retina_type, 'treatment' : treatment,                            'stimulus_type' : stimulus_type}

        # generate a Retina object if one has been specified
        if retina:
            self.retina_name = retina
            self.retina = Retina(self.experiment['retina_type'], experiment = self.experiment)

        # initialize the variables needed to run the experiment
        self.end_experiment = False


    def run_experiment(self, duration = None):
        '''
        Controls the entire simulation according to the parameters selected by the User.
        '''
        # **********************************     initialize model    ***************************************
        global tick_rate, ticks, last_tick

        if duration:
            self.early_end = duration * tick_rate
        else:
            self.early_end = None
        print('early end:', self.early_end)
        # generate the stimulus file (with User input?)
        stimulus = {'stimulus_type':'moving_bar', 'start_time':1, 'angles':[math.pi/2], 'widths':[100], 'lifespans':[1202.8 / 100],                     'speeds':[100], 'intensities':[100], 'wavelengths':['white'], 'delay':0.5, 'pinhole':0}

        # generate the mea file (with User input?)
        mea = {'channel_map':channel_map}

        # designate the channels to be used (with User input)
        channels = [27, 34, 4, 57]

        # add neurons to the retina
        neurons_per_channel = 1
        for channel in channels:
            for n in range(neurons_per_channel):
                self.retina.add_neuron(channel, neuron_type = 'RGC', subtype = 'type1')

        # initialize the stimulus
        self.stimulus = Moving_Bar(stimulus, channels, mea)


        # ***********************************     run simulation      *************************************

        while not self.end_experiment:

            # increment the ticks
            ticks += 1

            if ticks%1000 == 0:
                print(ticks, 'ticks')

            # update stimulus (this call to the Stimulus will update the neurons on any change in the stimulus.
            # It also returns a new value to end_experiment after the last stimulus is complete)
            self.end_experiment = self.stimulus.update_stimulus(self.retina)

            # check for early end to experiment
            if self.early_end and ticks >= self.early_end:
                self.end_experiment = True

            # update neurons
            for unit in range(len(self.retina.neurons)):

                # prompt neurons to udpate themselves and trigger themselves to spike
                self.retina.neurons[unit].update_neuron(self)

        # should store all data in a file

print('tick_rate', tick_rate, 'ticks', ticks, 'last_tick', last_tick)


# In[148]:

for n in range(len(exp_1.retina.neurons)):
    print('neuron', exp_1.retina.neurons[n], 'channel', exp_1.retina.neurons[n].channel, exp_1.retina.neurons[n].spike_train)


# In[147]:

exp_1 = Experiment(retina = 'E1_R1', retina_type = 'wild_type', treatment = None, stimulus_type = 'moving_bar')
exp_1.run_experiment(20)

# analyze data
#for unit in range(len(E1_R2.neurons)):
#    print(E1_R2.neurons[unit], ':', len(E1_R2.neurons[unit].get_spike_train()), E1_R2.neurons[unit].get_spike_train()) # create data list for spike plotter
data = []
fig = plt.figure(figsize = [10,6])
for unit in range(len(exp_1.retina.neurons)):
    spikes = np.array(exp_1.retina.neurons[unit].get_spike_train())
    data.append({'stimulus':{'lifespan':12.0}, 'spikes':spikes})
ax = plt.subplot(111)
axis_gen = axis_generator(ax)
plot_spike_trains(fig, axis_gen, data,prepend_start_time=1.0,append_lifespan=0.5,continuation=c_plot_solid, ymap=None)
# added by Larry for testing rf model outputs
plt.yticks([y for y in range(1,len(data)+1) if y%10==0])
plt.ylabel('Units')
plt.show()


# In[123]:

data = []
fig = plt.figure(figsize = [10,6])
for unit in range(len(E1_R2.neurons)):
    spikes = np.array(E1_R2.neurons[unit].get_spike_train())
    data.append({'stimulus':{'lifespan':0.5}, 'spikes':spikes})
ax = plt.subplot(111)
axis_gen = axis_generator(ax)
plot_spike_trains(fig, axis_gen, data,prepend_start_time=1.0,append_lifespan=2.0,continuation=c_plot_solid, ymap=None)
# added by Larry for testing rf model outputs
plt.yticks([y for y in range(1,len(data)+1) if y%10==0])
plt.ylabel('Units')
plt.show()


# In[54]:

test = Retina('model', experiment = 'test')
test.add_neuron(1, neuron_type = None, subtype = None)
for ticks in range(5000):
    if ticks < 1000:
        test.neurons[0].spiking(1)
    elif ticks < 4000:
        test.neurons[0].spiking(10)
    else:
        test.neurons[0].spiking(1)
print(len(test.neurons[0].get_spike_train()), test.neurons[0].get_spike_train())
spikes = np.array(test.neurons[0].get_spike_train())
data = [{'stimulus':{'lifespan':3.0}, 'spikes':spikes}]
fig = plt.figure()
ax = plt.subplot(111)
axis_gen = axis_generator(ax)
plot_spike_trains(fig, axis_gen, data,prepend_start_time=1,append_lifespan=1,
                      continuation=c_plot_solid, ymap=None)


plt.show()


# In[127]:

# (x,y)
channel_map = {
    1: (3,6),
    2: (3,7),
    3: (3,5),
    4: (3,4),
    5: (2,7),
    6: (2,6),
    7: (1,7),
    8: (2,5),
    9: (1,6),
    10: (0,6),
    11: (1,5),
    12: (0,5),
    13: (2,4),
    14: (1,4),
    15: (0,4),    # reference electrode
    16: (0,3),
    17: (1,3),
    18: (2,3),
    19: (0,2),
    20: (1,2),
    21: (0,1),
    22: (1,1),
    23: (2,2),
    24: (1,0),
    25: (2,1),
    26: (2,0),
    27: (3,3),
    28: (3,2),
    29: (3,0),
    30: (3,1),
    31: (4,1),
    32: (4,0),
    33: (4,2),
    34: (4,3),
    35: (5,0),
    36: (5,1),
    37: (6,0),
    38: (5,2),
    39: (6,1),
    40: (7,1),
    41: (6,2),
    42: (7,2),
    43: (5,3),
    44: (6,3),
    45: (7,3),
    46: (7,4),
    47: (6,4),
    48: (5,4),
    49: (7,5),
    50: (6,5),
    51: (7,6),
    52: (6,6),
    53: (5,5),
    54: (6,7),
    55: (5,6),
    56: (5,7),
    57: (4,4),
    58: (4,5),
    59: (4,7),
    60: (4,6)
}


# In[124]:

import functools
from functools import reduce
from typing import Dict, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Union
#from functional import f_filter, f_map, f_reduce
from scipy import signal
from glia.types import Unit
from tqdm import tqdm
import random    # added to Tyler's
#import analysis
from scipy import ndimage
import math


# *****************************************************************************************************************
#                               New Functions for Receptive Field Modeling 7/31/17
#
# *****************************************************************************************************************

def plot_firing_rate (spike_train):    # removed def plot_firing_rate (spike_train: SpikeTrain):
    """Take spike times of a particular spike unit and return a figure plotting
    1) firing rate vs spike times (green) 2) 1-Dimensional Gaussian filter of firing rate vs spike times (magenta)
    Firing rate estimated by the interspike interval.
    Requires 'from scipy import ndimage.'"""

    y = np.diff(spike_train)
    x = spike_train
    firing_rate = 1/y
    #add 0 to the end of the firing_rate array to account for last spike where no firing_rate is calculated
    #and to make x and y the same dimension
    firing_rate = np.append(firing_rate, 0)
    fig = plt.plot(x, firing_rate, color = 'green', linewidth=1)

    #sigma is standard deviation for Gaussian kernel
    sigma = 1.5
    x_g1d = ndimage.gaussian_filter1d(x, sigma)
    y_g1d = ndimage.gaussian_filter1d(firing_rate, sigma)
    mean_rate, st_dev = np.mean(y_g1d), np.std(y_g1d)

    fig = plt.plot(x_g1d, y_g1d, 'magenta', linewidth=1)

    plt.show()

def earliest_response_time(spike_train):
    '''Takes a list of spike times, estimates firing rates, then determines the earliest time
    where the firing rate is more than + 1 std greater than the mean firing rate.
    The spike_train and firing rates are smoothed with a Gaussian filter to reduce noise.

    spike_train: a list (or 1d np.array) of spike times (floats)

    Returns: response time (float): the earliest time when firing rate increases substanially
             over the mean firing rate, or None if no firing rates meet the criterion.
    '''

    y = np.diff(spike_train)
    x = spike_train
    firing_rate = 1/y
    #add 0 to the end of the firing_rate array to account for last spike where no firing_rate is calculated
    #and to make x and y the same dimension
    firing_rate = np.append(firing_rate, 0)
    fig = plt.plot(x, firing_rate, color = 'green', linewidth=1)

    #sigma is standard deviation for Gaussian kernel
    sigma = 1.5
    x_g1d = ndimage.gaussian_filter1d(x, sigma)
    y_g1d = ndimage.gaussian_filter1d(firing_rate, sigma)

    # determine threshold for selecting the earliest response time
    mean_rate, st_dev = np.mean(y_g1d), np.std(y_g1d)
    threshold = mean_rate +  st_dev

    # find and return the earliest response time, or None if none is found
    for t in range(len(y_g1d)):
        if y_g1d[t] > threshold:
            return x_g1d[t]
    else:
        return None

def read_stimulus_dictionary(stimulus):
    '''
    Reads relevant data from the stimulus dictionary.

    stimulus:   dict: keys are attributes of the stimulus; values are the values of the attributes

    The following values are read from the stimulus dictionary and returned:

    stimulus_type: str: the type of stimulus (e.g. moving bar, grating, checkerboard, etc.)
    start_time:  int: the time, in secs, when the first stimulus begins
    num_trials:  int: the # of times each stimulus is repeated
    num_angles:  int: the number of angles (12 in the actual experiments)
    angles:      nd.array of floats with shape(num_units, num_angles): each bar's angle of origin in radians
    num_widths:  int: the number of different bar widths in the stimulus
    widths:      list: contains int values of the different bar widths (in pixels)
    lifespans:   list: (of lists) containing floats - the duration of each presentation of a stimulus (in seconds)
    num_speeds:  int: the number of different bar speeds in the stimulus
    speeds:      list: the integer values of the different bar speeds (in pixels/sec)
    num_intensities: int: the number of different light intensities in the stimulus
    intensities: list: integer values of light intensities (units arbitrary)
    num_colors:  int: the number of different wavelengths presented by the stimulus
    wavelengths: list: int values of the wavelengths (in angstroms), or the str 'white' for full-spectrum
    delay:       float: the time between presentation of different stimuli in sec. (default = 0.5 sec)
    image_width: float: the horizontal distance across the projected image in microns (ARBITRARY default = 3,000 microns)
    image_height:float: the vertical distance across the projected image in microns (ARBITRARY default = 2,000 microns)
                        image_width and image_height are used to calculate the distance between the bar and a neuron
    pixel_size:  float: the average size (in microns) of a single pixel of the stimulus image; ARBITRARY default = 10 microns)
    pinhole:     float: the pinhole setting of projector, which may affect image resolution
    '''

    stimulus_type = stimulus.get('stimulus_type', 'moving_bar')
    start_time = stimulus.get('start_time', 0.5)
    num_trials = stimulus.get('num_trials', 1)
    angles = stimulus.get('angles')
    if angles == None:
        raise ValueError ('No angles provided in stimulus dictionary')
    num_angles = len(angles)
    widths = stimulus.get('widths', [10])
    if widths == None:
        raise ValueError ('No widths provided in stimulus dictionary')
    num_widths = len(widths)
    lifespans = stimulus.get('lifespans')
    if lifespans == None:
        raise ValueError ('No lifespans provided in stimulus dictionary')
    speeds = stimulus.get('speeds')
    if speeds == None:
        raise ValueError ('No speeds provided in stimulus dictionary')
    num_speeds = len(speeds)
    intensities = stimulus.get('intensities')
    if intensities == None:
        raise ValueError ('No light intensities provided in stimulus dictionary')
    num_intensities = len(intensities)
    wavelengths = stimulus.get('wavelengths', ['white'])
    num_colors = len(wavelengths)
    delay = stimulus.get('delay', 0.5)
    image_width = stimulus.get('image_width', 3000)
    image_height = stimulus.get('image_height', 2000)
    pixel_size = stimulus.get('pixel_size', 10.0)
    pinhole = stimulus.get('pinhole', 0)

    stim_readout = stimulus_type, start_time, num_trials, num_angles, angles, num_widths, widths, lifespans,     num_speeds, speeds, num_intensities, intensities, num_colors, wavelengths, delay,     image_width, image_height, pixel_size, pinhole

    return stim_readout


def read_mea_dictionary(mea):
    '''
    Reads relevant data from the mea dictionary.

    mea:         dict: keys are attributes of the stimulus; values are the values of the attributes

    The following values are read from the mea dictionary and returned:

    channel_map: dict: keys: channel_ids; values: int tuples of (x,y) positions within the mea [(0,0) = upper left]
    channel_spacing: int: the distance between channels of the mea, in microns. default is 200 microns
    offset_angle:float: the angle (an radians) required to transform the stimulus image coordinates to match the mea coordinates
    offset_x:    float: horizontal offset (in microns) required to transform the stimulus image coordinates to match mea coordinates
    offset_y:    float: vertical offset (in microns) required to transform the stimulus image coordinates to match mea coordinates
    '''

    channel_map = mea.get('channel_map')
    if channel_map == None:
        raise ValueError('No channel_map provided in mea dictionary')
    else:
        channel_spacing = mea.get('channel_spacing', 200)
        offset_angle = mea.get('offset_angle', 0)
        offset_x = mea.get('offset_x', 0)
        offset_y = mea.get('offset_y', 0)

    return channel_map, channel_spacing, offset_angle, offset_x, offset_y

def create_DATh_dictionary(stimulus, channels, mea):
    '''
    Creates a dictionary of D(A,theta) values which are used to find the distance between a neuron and
    the moving bar stimulus. D(theta) is the distance between the origin (i.e. center of the stimulus image) and
    a moving bar with angle = theta at time t = 0 (this is one-half the distance travelled by the front of the
    moving bar). D(A, theta) is the distance between a given unit (specifically, the channel recording that unit)
    and the moving bar with angle = theta at time t = 0.
    DATh_dict is a dictionary of dictionaries containing floats which are the D(A, theta) values for each channel.

    stimulus:   dict: keys are attributes of the stimulus; values are the values of the attributes
    channels:   list: list of the channel numbers recording data for experiment
    mea:        dict: keys are attributes of the mea; values are the values of the attributes
    '''
    #
    # ********* pinhole setting may affect the determination of bar location at any given t?? ***********

    # read data from the stimulus dictionary
    stimulus_type, start_time, num_trials, num_angles, angles, num_widths, widths, lifespans,     num_speeds, speeds, num_intensities, intensities, num_colors, wavelengths, delay,     image_width, image_height, pixel_size, pinhole = read_stimulus_dictionary(stimulus)

    # read data from the mea dictionary
    channel_map, channel_spacing, offset_angle, offset_x, offset_y = read_mea_dictionary(mea)

    if not channels:
        channels = [c for c in range(1, len(channel_map) + 1)]

    # Create DTh_dict, a dictionary containing the D(theta) values for each angle. First, calculate PL(theta),
    # the path length that the leading edge of the moving bar will travel for each angle, theta.
    # This does not include the extra distance traveled due to the bar width. PL(theta) is exactly twice D(theta),
    # the distance from the center of the stimulus image to the leading edge of the moving bar at time t = 0.
    DTh_dict = {}
    # calculate PL(theta) for each angle
    for theta in angles:
        if theta <= math.pi / 2:            # theta in quadrant I
            PLTh = image_height * math.sin(theta) + image_width * math.cos(theta)
        elif theta <= math.pi:              # theta in quadrant II
            PLTh = image_height * math.sin(math.pi - theta) + image_width * math.cos(math.pi - theta)
        elif theta <= 3 * math.pi / 2:      # theta in quadrant III
            PLTh = image_height * math.sin(theta - math.pi) + image_width * math.cos(theta - math.pi)
        else:                               # theta in quadrant IV
            PLTh = image_height * math.sin(2 * math.pi - theta) + image_width * math.cos(2 * math.pi - theta)
        # D(theta) for each angle = PL(theta) / 2
        DTh_dict[theta] = PLTh / 2

    # create DATh_dict
    DATh_dict = {}
    for channel in channels:
        DATh_dict[channel] = {}

        # convert channel #'s to position on mea grid using channel_map.
        channel_x, channel_y = channel_map[channel]
        channel_x, channel_y = 2 * channel_x - 7, 7 - 2 * channel_y
        # scale (x,y) values according to the distance between the mea channels (channels are 2 ticks apart)
        channel_x *= channel_spacing / 2
        channel_y *= channel_spacing / 2
        # translate each point to account for the mea being placed off-center, if necessary
        channel_x += offset_x
        channel_y += offset_y

        # find alpha, the angle of the vector A from the center of the stimulus image to the current channel (A)
        # using (x,y) in unit_locs
        if channel_x >= 0.0:
            if channel_y >= 0.0:     # channel A in quadrant I
                alpha = math.atan(channel_y / channel_x)
            else:                    # channel A in quadrant IV
                alpha = 2 * math.pi - math.atan(channel_y / channel_x)
        elif channel_y >= 0.0:       # channel A in quadrant II
            alpha = math.pi - math.atan(channel_y / channel_x)
        else:                        # channel A in quadrant III
            alpha = math.pi + math.atan(channel_y / channel_x)
        # rotate alpha to account for the mea being misaligned to the coordinate system of the stimulus image,
        # if necessary
        alpha += offset_angle

        # for each channel, find D(A,theta), where D(A, theta) is the distance from the channel (A)
        # to the bar's leading edge at time t = 0. D(A, theta) will be equal to D(theta) - the projection of vector A
        # onto D(theta).
        # In the figure, theta is the angle of the vector D(theta) from the origin to the moving bar,
        # alpha is the angle of the vector A to the channel (A), and psi is the difference (alpha - theta).
        #   . . . . . . . . . . . . . . . . . . . . . . # . . . . . . . . . . . . . .
        #   . . . . . . . . . . . . . . . . . . D(A, theta) . . . . . . . . . . . . .
        #   . . . . . . . . . . . . . . . . . . . . / . . . # . . . . . . . . . . . .
        #   . . . . . . . . . . . . . . . . . . . / . . . . . # . . . . . . . . . . .
        #   . . . . . . . . . . . . . . . . . . / . . . . .D(theta) . . . . . . . . .
        #   . . . . . . . . . . . . . . . . . / . . . . . . . / . # . . . . . . . . .
        #   . . . . . . . . . . . . . . . . / . . . . . . . / . . . # .moving bar @ t = 0
        #   . . . . . . . . . . . . . .(A).*. . . . . . . / . . . . . # . . . . . . .
        #   . . . . . . . . . . . . . (x,y) \ . psi . . / . . . . . . . # . . . . . .
        #   . . . . . . . . . . . . . . . . .\. . . . / . . theta . . . . # . . . . .
        #   . . . . . . . . . . . . . . . . . \ . alpha . . . . . . . . . . # . . . .
        #   . . . . . . . . . . . . . . . . . .\. / . . . . . . . . . . . . . # . . .
        #   . . . . . . . . . . . . . . . . . . + ------------------------------#----
        #   . . . . . . . . . . . . . . . . (0,0) . . . . . . . . . . . . . . . . # .
        #   . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . #
        #   . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        for theta in angles:
            # find psi, the angle between the vector to channel A and the vector D(theta)
            psi = alpha - theta

            # find k, the scalar of the projection of the vector A onto the vector D(theta)
            # k = (||A||*cos(psi)) / ||D(theta)|| where ||A|| = sqrt(x^2 + y^2).
            k = (math.sqrt(channel_x**2 + channel_y**2) * math.cos(psi)) / DTh_dict[theta]

            # Finally, D(A, theta) = D(theta) * (1 - k)
            DATh_dict[channel][theta] =  round(DTh_dict[theta] * (1 - k))

    return DATh_dict


# In[ ]:
