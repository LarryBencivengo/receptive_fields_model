
# coding: utf-8

# In[1]:

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
    15: (0,4),
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


# In[18]:

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


def random_unit(total_time, retina_id, channel, unit_num):
    spike_train = []
    spike = np.random.random()
    while spike < total_time:
        spike_train.append(spike)
        spike += np.random.random()

    unit = Unit(retina_id, channel, unit_num, np.array(spike_train))
    return unit

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
    print(y_g1d)
    print(mean_rate)
    print(st_dev)
    print(earliest_response_time(spike_train))

    fig = plt.plot(x_g1d, y_g1d, 'magenta', linewidth=1)

    return fig

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


# *************************************** plot_receptive_fields **************************************************
#                                                7/25/17
# latest function to plot receptive fields
#
# ****************************************************************************************************************

def plot_receptive_fields(radii, num_units = None, num_angles = None, angles = None,                          num_rows = None, num_cols = 6, image_height = 10, image_width = 15, channel_spacing = 200):
    '''
    Takes a  list of radii of receptive fields from a moving bar stimulus then creates a plot showing
    the receptive fields of all units.

    radii:       dictionary: for each unit, a list of floats - the radius (in microns) of the receptive field at each angle tested
    num_units:   int: optional parameter extracted from the first dimension of spikeTrains
    num_angles:  int: optional parameter extracted from the 2nd dimension of spikeTrains (usually 12)
    angles:      nd.array of floats with shape(num_units, num_angles): optional parameter with each bar's angle in radians
    num_rows:    int: optional parameter calculated from num_units and num_cols
    num_cols:    int: optional parameter (default = 6)
    image_width: float: the horizontal distance across the projected image (ARBITRARY default = 15 mm)
    image_height:float: the vertical distance across the projected image (ARBITRARY default = 10 mm)
                 image_width and image_height are used to calculate the distance between the bar
                 and a unit when it begins spiking
    channel_spacing: int: the distance between channels of the mea, in microns. default is 200 microns

    returns: fig -> matplotlib.###.### instance containing num_units subplots arranged in a
                    num_rows by num_cols grid. Each subplot shows the receptive field of one unit.
    '''

    # ***************************** create the figure with appropriate subplots to return ******************************

    # calculate numer of rows needed for the # of columns specified, if not provided
    if not num_rows:
        num_rows = int(num_units // num_cols + math.ceil((num_units % num_cols) / num_cols))


    unit_num = 0         # initialize counter in order to label subplots by unit number

    plt.figure(figsize = (10, 12))

    for row in range(num_rows):
        for col in range(num_cols):
            if unit_num < num_units:     # uses Gridspec and subplot2grid to create gridded array
                ax = plt.subplot2grid((num_rows, num_cols), (row, col), projection='polar')
                y = radii[unit_num, :]       # select the row of the matrix corresponding to this unit
                ax.plot(theta, y)
                ax.set_rticks([])        # turn off radial ticks and labels
                ax.set_xticks([0, np.pi /2, np.pi, 3 * np.pi /2])
                ax.tick_params(axis='x', labelsize='x-small')  # get 4 small angle labels
                #ax.set_rmax(r.max())     # scale all subplots to largest receptive field
                ax.set_rmax(image_diam)   # scale all subplots to the image size
                ax.set_title(str(unit_num) + '      ', va = 'top', ha = 'right')
                ax.grid(True)
                unit_num += 1            # increment unit counter
            else:
                break

    plt.tight_layout(w_pad=1.0)          # creates space between subplots

    return fig



# *************************************** map_receptive_fields **************************************************
#                                                7/25/17
# latest function to map receptive fields
#
# ****************************************************************************************************************

def map_receptive_fields(spike_trains, stimulus = None, num_units = None, unit_locs = None,                           num_angles = None, angles = None, num_speeds = None, speeds = None,                           num_widths = None, widths = None, lifespans = None,                           num_rows = None, num_cols = 6, image_height = 10000, image_width = 15000,                           pinhole = 0, channel_spacing = 200, offset_angle = 0.0, offset_x = 0.0, offset_y = 0.0):
    '''
    Takes a list of spike_trains from a moving bar stimulus experiment, extracts the earliest_response_time
    for each unit at each angle, then returns a dictionary of the radius of each unit's receptive field at each angle

    spike_trains: a list of lists of floats: the spike_times each unit for each moving bar stimulus
    stimulus:    dictionary: "optional" parameter which provides info. on bar angles, widths, speeds, lifespans, etc.
    num_units:   int: optional parameter extracted from the first dimension of spikeTrains
    unit_locs:   list of ints: channel # of each unit. Use channel_map to find grid location of each unit on mea.
    num_angles:  int: optional parameter extracted from the 2nd dimension of spikeTrains (usually 12)
    angles:      nd.array of floats with shape(num_units, num_angles): optional parameter with each bar's angle in radians
    num_speeds:  int: optional parameter extracted from stimulus file
    speeds:      nd.array of ints (bar speeds in pixels/sec): optional parameter extracted from stimulus file
    lifespans:   nd.array of floats (lifespans in seconds): optional parameter extracted from stimulus file
    num_widths:  int: number of different bar widths
    widths:      nd.array of ints (width of each bar in pixels)
    num_rows:    int: optional parameter calculated from num_units and num_cols
    num_cols:    int: optional parameter (default = 6)
    image_width: float: the horizontal distance across the projected image in microns (ARBITRARY default = 15,000 microns)
    image_height:float: the vertical distance across the projected image in microns (ARBITRARY default = 10,000 microns)
                 image_width and image_height are used to calculate the distance between the bar
                 and a unit when it begins spiking
    pinhole:     float: the pinhole (fstop?) setting of projector, which may affect image resolution
    channel_spacing: int: the distance between channels of the mea, in microns. default is 200 microns
    offset_angle:float: the angle required to transform the stimulus image coordinates to match the mea coordinates
    offset_x:    float: horizontal offset required to transform the stimulus image coordinates to match mea coordinates
    offset_y:    float: vertical offset required to transform the stimulus image coordinates to match mea coordinates

    returns: rTh_dict -> dictionary containing the radius of each unit's receptive field at each angle tested.
                        Key: unit_id; Value: list of floats - the radius of receptive field for each angle
    '''

    # spikeTrains may be replaced by a list of unit objects (unit_list)? containing the spikeTrains, channel_ID, etc.


    def read_stimulus_file(spike_trains, stimulus):
        '''
        Reads out the stimulus information required to map each unit's receptive field.

        XXXXXXXXXXXXXX - more info. later

        return: tuple -> (num_units, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole)
        '''

        # construct phony stimulus dictionary, if indicated.
        # stimulus should include: num_angles, angles [for each spikeTrain of each unit], num_speeds,
        #                          speeds [for each spikeTrain of each unit], lifespans [for each spikeTrain of each unit],
        #                          and pinhole setting of projector
        if stimulus == 'create_phony':
            # create the phony stimulus file
            stimulus = {}
            stimulus['experiment'] = 'test'
            stimulus['num_angles'] = 12
            stimulus['num_speeds'] = 1
            stmulus['num_widths'] = 1
            angles_list = []
            speeds_list = []
            lifespans_list = []
            widths_list = []
            # generate the info.
            for unit in range(num_units):
                angles_list.append([unit])   # this does nothing at the moment
                speeds_list.append([unit])   # this does nothing at the moment
                lifespans_list.append([unit])# this does nothing at the moment
                widths_list.append([unit])   # this does nothing at the moment
                for st in range(stimulus['num_angles']):
                    angles_list[unit].append(st)
                    speeds_list[unit].append(100)      # only one speed currently in test
                    lifespans_list.append(1202.8 / speeds_list[unit][st])
                    widths_list[unit].append
            # create remainder of the phony stimulus file
            stimulus['angles'] = angles_list
            stimulus['speeds'] = speeds_list
            stimulus['lifespans'] = lifespans_list
            stimulus['pinhole'] = None

        # read the stimulus file, if provided (including the phony file)
        if stimulus:
            # determine num_units, if not provided
            if not num_units:
                num_units = spike_trains.shape[0]
            if not angles:
                angles = stimulus['angles']
            if not speeds:
                speeds = stimulus['speeds']
            if not lifespans:
                lifespans = stimulus['lifespans']
            if not pinhole:
                pinhole = stimulus['pinhole']
        # otherwise generate the required stimulus information for testing purposes
        else:
            # determine num_units, if not provided
            if not num_units:
                num_units = spike_trains.shape[0]

            # unit_locs must be provided to this function, but, for now, it sets arbitrary locations for each unit
            # in order to test the mapping function's ability to find the correct receptive fields for units
            # near different channels of the mea.
            # channel_map is a dictionary giving a tuple (column, row) for each channel, with (0,0) indicating the upper left
            # of the mea, where no electrode is present. This tuple can be used to determine a channel's location relative
            # to the origin (i.e. center of the stimulus image).
            if not unit_locs:
                unit_locs = []
                i = 1
                for unit in range(num_units):  # randomly assign a channel to each unit
                    unit_locs.append(i)
                    if random.random() >= 0.5:
                        i += 1
                        if i > 60:
                            i = 60

            # determine num_angles, if not provided, from the stimulus file.
            # currently uses the number of spikeTrains per unit to determine num_angles
            # (assumes each spikeTrain is the sole data set for a unique angle.)
            # This must be fixed to be able to take an average of different trials at various speeds, widths, etc. for each angle
            if not num_angles:
                num_angles = spikeTrains.shape[1]

            # determine the angles, if not provided, from the stimulus file.  Angles are in radians.  Origin is the center
            # of the video image; the reference direction is toward the right along the horizontal axis.
            # Currently uses the number of spikeTrains per unit to determine num_angles
            # (assumes each spikeTrain is the sole data set for a unique angle.)
            # This must be fixed to be able to take an average of different trials at various speeds, widths, etc. for each angle
            if not angles:
                angles = np.full((num_units, num_angles), 0)
                angle_list = np.linspace(0.1, 2*np.pi-0.1, 12)  # generates 12 evenly spaced angles; these are NOT correct
                for unit in range(num_units):
                    for angle in range(num_angles):
                        angles[unit][angle] = angle_list[angle]

            # determine the bar speeds (including num_speeds) from the stimulus file, if provided. Creates nd.array of bar speeds
            # currently ARBITRARILY sets all bar speeds to 100 pixels/sec

            if not num_speeds:
                # num_speeds = spikeTrains.shape[2]
                num_speeds = num_angles

            if not speeds:
                speeds = np.full((num_units, num_speeds), 100)   # must eventually extract this info. from stimulus file


            # determine the bar widths from the stimulus file, if not provided.
            if not num_widths:
                num_widths = num_angles
            # create widths, a list of lists containing the width of each bar for each unit (currently set to 0 pixels)
            if not widths:
                widths = np.full((num_units, num_widths), 0)

            # determine the lifespan of each bar from the stimulus file, if provided.  Creates an nd.array of
            # lifespans from the speeds using the following relationship: lifespan = 1202.8 / speed
            # NB: lifespan actually depends on both speed and bar width!!

            if not lifespans:
                lifespans = np.full((num_units, num_angles), 0)
                for unit in range(num_units):
                    s = 0
                    for speed in speeds[unit][ : ]:
                        lifespans[unit][s] = 1202.8 / speed
                        s += 1

            # determine pinhole setting from stimulus file
            if not pinhole:
                pass     # determine pinhole setting from stimulus file

        #return num_units, unit_locs, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole
        return None, None, None, None, None, None, None, None, None, None



    # read stimulus info from stimulus file
    *num_units, unit_locs, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole     = read_stimulus_file(spike_trains, stimulus)






    # convert channel # to (x,y) coordinates on mea grid that can be used to determine location within stimulus image
    for unit in range(len(unit_locs)):

        # convert mea grid location to (x,y) coordinates within the stimulus image with geometric translation
        unit_locs[unit] = ()


    def create_DATh_library(channel_map, num_angles, angles, image_height, image_width, pinhole, channel_spacing,                             offset_angle, offset_x, offset_y):
        '''
        Creates a dictionary of D(A,theta) values which are used to determine the radius of a unit's receptive field for each
        bar-angle, theta. D(theta) is the distance between the origin (i.e. center of the stimulus image) and a moving bar
        with angle = theta at time t = 0 (this is one-half the distance travelled by the front of the moving bar).
        D(A, theta) is the distance between a given unit (specifically, the channel recording that unit)
        and the moving bar with angle = theta at time t = 0.
        DATh_dict is a dictionary of dictionaries containing floats which are the D(A, theta) values for each channel.

        channel_map: lib: keys: channel_ids; values: int tuples of (x,y) positions within the mea [(0,0) = upper left]
        num_angles:  int: optional parameter extracted from the 2nd dimension of spikeTrains (usually 12)
        angles:      nd.array of floats with shape(num_units, num_angles): optional parameter with each bar's angle in radians
        image_width: float: the horizontal distance across the projected image in microns (ARBITRARY default = 15,000 microns)
        image_height:float: the vertical distance across the projected image in microns (ARBITRARY default = 10,000 microns)
                     image_width and image_height are used to calculate the distance between the bar
                     and a unit when it begins spiking
        pinhole:     float: the pinhole (fstop?) setting of projector, which may affect image resolution
        channel_spacing: int: the distance between channels of the mea, in microns. default is 200 microns
        offset_angle:float: the angle (an radians) required to transform the stimulus image coordinates to match the mea coordinates
        offset_x:    float: horizontal offset (in microns) required to transform the stimulus image coordinates to match mea coordinates
        offset_y:    float: vertical offset (in microns) required to transform the stimulus image coordinates to match mea coordinates
        '''
        #
        # ********* pinhole setting may affect the determination of bar location at any given t?? ***********

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

        DATh_dict = {}
        for channel in range(1, len(channel_map) + 1):
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
                    alpha = math.atan(channel_y, channel_x)
                else:                    # channel A in quadrant IV
                    alpha = 2 * math.pi - math.atan(channel_y, channel_x)
            elif channel_y >= 0.0:       # channel A in quadrant II
                alpha = math.pi - math.atan(channel_y, channel_x)
            else:                        # channel A in quadrant III
                alpha = math.pi + math.atan(channel_y, channel_x)
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
                DATh_dict[channel, theta] =  DTh_dict[theta] * (1 - k)

        return DATh_dict

    def generate_map_dictionary(spike_trains, DATh_dict, num_units, unit_locs, num_angles, angles, num_speeds, speeds,                                 num_widths, widths, lifespans, pinhole, channel_spacing):
        '''
        Takes the spike_trains and determines the earliest_response_time (t) for each unit at each trial. Uses the
        D(A, theta) values calculated previously to determine d(theta), the distance to the moving bar at time t.
        Averages all the values for each angle to determine r(theta), the radius of the unit's receptive field
        at the angle theta for all angles tested.

        spike_trains: a list of lists of floats: the spike_times each unit for each moving bar stimulus
        num_units:   int: optional parameter extracted from the first dimension of spikeTrains
        unit_locs:   list of ints: channel # of each unit. Use channel_map to find grid location of each unit on mea.
        num_angles:  int: optional parameter extracted from the 2nd dimension of spikeTrains (usually 12)
        angles:      nd.array of floats with shape(num_units, num_angles): optional parameter with each bar's angle in radians
        num_speeds:  int: optional parameter extracted from stimulus file
        speeds:      nd.array of ints (bar speeds in pixels/sec): optional parameter extracted from stimulus file
        lifespans:   nd.array of floats (lifespans in seconds): optional parameter extracted from stimulus file
        num_widths:  int: number of different bar widths
        widths:      nd.array of ints (width of each bar in pixels)
        pinhole:     float: the pinhole (fstop?) setting of projector, which may affect image resolution
        channel_spacing: int: the distance between channels of the mea, in microns. default is 200 microns

        returns:     rTh_dict: dictionary: Keys: unit_ids; Values: list of the radius (in microns) of receptive field at each angle

        # ************ currently assumes each spikeTrain is the sole data set for a unique angle. ********************
        # This must be fixed to take an average of different trials at various speeds, bar widths, etc. for each angle
        '''

        # **************************** pinhole value may be needed to find correct r(theta) values !! *****************

        response_times = nd.full((num_units, num_angles), 0)
        for unit in range(num_units):
            a = 0
            # currently, there is only one spikeTrain per angle per unit.  FIX THIS!!
            for spikeTrain in spike_trains[unit, : ]:
                response_times[unit, a] = earliest_response_time(spike_train)
                a += 1






    # *********************** generate fake data using artifical units, then plot receptive fields ********************
num_units = 47
num_cols = 6         # choose based on desired size/spread of plots


spikeTrains = np.array(num_units, 12)
my_unit = random_unit(6.0, 'E1_R1', 1, 0) # params: (total_time, retina_id, channel, unit_num)
# my_fig = plot_firing_rate(my_unit.spike_train)


plt.show()



# In[19]:

stimulus = {}
if stimulus:
    print('there is a stimulus file!')
else:
    print('no file')


# In[45]:

#return num_units, unit_locs, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole
num_units = None
unit_locs = None
num_angles = None
angles = None
num_speeds = None
speeds = None
lifespans = None
num_widths = None
widths = None
pinhole = None

def read_stimulus_file(spike_trains, stimulus, num_units = None):
    '''
    Reads out the stimulus information required to map each unit's receptive field.

    XXXXXXXXXXXXXX - more info. later

    return: tuple -> (num_units, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole)
    '''

    # construct phony stimulus dictionary, if indicated.
    # stimulus should include: num_angles, angles [for each spikeTrain of each unit], num_speeds,
    #                          speeds [for each spikeTrain of each unit], lifespans [for each spikeTrain of each unit],
    #                          and pinhole setting of projector
    if stimulus == 'create_phony':
        # create the phony stimulus file
        stimulus = {}
        stimulus['experiment'] = 'test'
        stimulus['num_angles'] = 12
        stimulus['num_speeds'] = 1
        stimulus['num_widths'] = 1
        angles_list = []
        speeds_list = []
        lifespans_list = []
        widths_list = []
        # generate the info.
        if not num_units:
            num_units = spike_trains.shape[0]
        for unit in range(num_units):
            angles_list.append([])   # this does nothing at the moment
            speeds_list.append([])   # this does nothing at the moment
            lifespans_list.append([unit])# this does nothing at the moment
            widths_list.append([unit])   # this does nothing at the moment
            for st in range(stimulus['num_angles']):
                angles_list[unit].append(st)       # this does nothing at the moment
                speeds_list[unit].append(100)      # only one speed currently in test
                lifespans_list.append(1202.8 / speeds_list[unit][st])
                widths_list[unit].append
        # create remainder of the phony stimulus file
        stimulus['angles'] = angles_list
        stimulus['speeds'] = speeds_list
        stimulus['lifespans'] = lifespans_list
        stimulus['pinhole'] = None

    # read the stimulus file, if provided (including the phony file)
    if stimulus:
        # determine num_units, if not provided
        if not num_units:
            num_units = spike_trains.shape[0]
        if not angles:
            angles = stimulus['angles']
        if not speeds:
            speeds = stimulus['speeds']
        if not lifespans:
            lifespans = stimulus['lifespans']
        if not pinhole:
            pinhole = stimulus['pinhole']
    # otherwise generate the required stimulus information for testing purposes
    else:
        # determine num_units, if not provided
        if not num_units:
            num_units = spike_trains.shape[0]

        # unit_locs must be provided to this function, but, for now, it sets arbitrary locations for each unit
        # in order to test the mapping function's ability to find the correct receptive fields for units
        # near different channels of the mea.
        # channel_map is a dictionary giving a tuple (column, row) for each channel, with (0,0) indicating the upper left
        # of the mea, where no electrode is present. This tuple can be used to determine a channel's location relative
        # to the origin (i.e. center of the stimulus image).
        if not unit_locs:
            unit_locs = []
            i = 1
            for unit in range(num_units):  # randomly assign a channel to each unit
                unit_locs.append(i)
                if random.random() >= 0.5:
                    i += 1
                    if i > 60:
                        i = 60

        # determine num_angles, if not provided, from the stimulus file.
        # currently uses the number of spikeTrains per unit to determine num_angles
        # (assumes each spikeTrain is the sole data set for a unique angle.)
        # This must be fixed to be able to take an average of different trials at various speeds, widths, etc. for each angle
        if not num_angles:
            num_angles = spikeTrains.shape[1]

        # determine the angles, if not provided, from the stimulus file.  Angles are in radians.  Origin is the center
        # of the video image; the reference direction is toward the right along the horizontal axis.
        # Currently uses the number of spikeTrains per unit to determine num_angles
        # (assumes each spikeTrain is the sole data set for a unique angle.)
        # This must be fixed to be able to take an average of different trials at various speeds, widths, etc. for each angle
        if not angles:
            angles = np.full((num_units, num_angles), 0)
            angle_list = np.linspace(0.1, 2*np.pi-0.1, 12)  # generates 12 evenly spaced angles; these are NOT correct
            for unit in range(num_units):
                for angle in range(num_angles):
                    angles[unit][angle] = angle_list[angle]

        # determine the bar speeds (including num_speeds) from the stimulus file, if provided. Creates nd.array of bar speeds
        # currently ARBITRARILY sets all bar speeds to 100 pixels/sec

        if not num_speeds:
            # num_speeds = spikeTrains.shape[2]
            num_speeds = num_angles

        if not speeds:
            speeds = np.full((num_units, num_speeds), 100)   # must eventually extract this info. from stimulus file


        # determine the bar widths from the stimulus file, if not provided.
        if not num_widths:
            num_widths = num_angles
        # create widths, a list of lists containing the width of each bar for each unit (currently set to 0 pixels)
        if not widths:
            widths = np.full((num_units, num_widths), 0)

        # determine the lifespan of each bar from the stimulus file, if provided.  Creates an nd.array of
        # lifespans from the speeds using the following relationship: lifespan = 1202.8 / speed
        # NB: lifespan actually depends on both speed and bar width!!

        if not lifespans:
            lifespans = np.full((num_units, num_angles), 0)
            for unit in range(num_units):
                s = 0
                for speed in speeds[unit][ : ]:
                    lifespans[unit][s] = 1202.8 / speed
                    s += 1

        # determine pinhole setting from stimulus file
        if not pinhole:
            pass     # determine pinhole setting from stimulus file

    #return num_units, unit_locs, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole
    return None, None, None, None, None, None, None, None, None, None

# *********************** generate fake data using artifical units, then plot receptive fields ********************
num_units = 47

spikeTrains = []
for num in range(num_units):
    spikeTrains.append([])
    for st in range(12):
        spikeTrains[num].append(random_unit(6.0, 'E1_R1', 1, 0).spike_train)
stimulus = 'create_phony'

# read stimulus info from stimulus file
*num_units, unit_locs, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole = read_stimulus_file(np.array(spikeTrains), stimulus)

print(*num_units, unit_locs, num_angles, angles, num_speeds, speeds, lifespans, num_widths, widths, pinhole)


# In[6]:

print ("\(copyright symbol here\)")


# In[16]:

channel_x, channel_y = 1,1
offset_x, offset_y = 50,50
channel_spacing = 200
# convert channel #'s to position on mea grid using channel_map.
channel_x, channel_y = 2 * channel_x - 7, 7 - 2 * channel_y
# scale (x,y) values according to the distance between the mea channels (channels are 2 ticks apart)
channel_x *= channel_spacing / 2
channel_y *= channel_spacing / 2
# translate each point to account for the mea being placed off-center, if necessary
channel_x += offset_x
channel_y += offset_y
print('channel_x =', channel_x, 'channel_y =', channel_y)


# In[36]:

# *********************** generate fake data using artifical units, then plot receptive fields ********************
num_units = 47

spikeTrains = []
for num in range(num_units):
    spikeTrains.append([])
    for st in range(12):
        spikeTrains[num].append(random_unit(6.0, 'E1_R1', 1, 0).spike_train)
print(spikeTrains)


# In[35]:

print(np.shape(spikeTrains))


# In[84]:

Th = np.linspace(0, 2*math.pi-(2*math.pi/12), 12)
Th += (math.pi/12)
Th = list(Th)
X = 3000
Y = 2000
D_Th = []
for i in range(3):
    D_Th.append(Y*math.sin(Th[i])+X*math.cos(Th[i]))
for i in range(3,6):
    D_Th.append(Y*math.sin(math.pi-Th[i])+X*math.cos(math.pi-Th[i]))
for i in range(6,9):
    D_Th.append(Y*math.sin(Th[i]-math.pi)+X*math.cos(Th[i]-math.pi))
for i in range(9,12):
    D_Th.append(Y*math.sin(2*math.pi-Th[i])+X*math.cos(2*math.pi-Th[i]))

print(Th, D_Th)


# In[91]:

#fig = plt.figure(size = (8,3)
plt.plot(Th,D_Th)
plt.xticks(Th)

plt.show()


# In[ ]:
