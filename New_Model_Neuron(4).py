
# coding: utf-8

# In[377]:

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
                    self.next_update = last_tick
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


# In[378]:

exp_1 = Experiment(retina = 'E1_R1', retina_type = 'wild_type', treatment = None, stimulus_type = 'moving_bar')
exp_1.run_experiment(20)


# In[379]:

for n in range(len(exp_1.retina.neurons)):
    print(exp_1.retina.neurons[n].spike_train)


# In[215]:

stimulus = {'stimulus_type':'moving_bar', 'start_time':5, 'angles':[math.pi/2], 'widths':[100], 'lifespans':[1202.8 / 100], 'speeds':[100], 'intensities':[100], 'wavelengths':['white'], 'delay':1.0, 'pinhole':0}
mea = {'channel_map':channel_map}
stim1 = Moving_Bar(stimulus, [26, 33, 4, 56], mea)
print(stim1.DATh_dict)


# In[202]:

testRun = Experiment('E1_R10', retina_type = 'wild_type', treatment = 'AMES')
print(testRun.id, testRun.experiment, testRun.retina.experiment)


# In[179]:

# initialize model (this normally also involves instantiating an Experiment object)
Neuron.id = 0
E1_R2 = Retina('wild_type')
num_units = 4
for unit in range(num_units):
    E1_R2.add_neuron(unit, neuron_type = 'RGC')
# define stimulus parameters (this normally involves instantiating a Stimulus object)
light_ON = False
intensity = 0
distance = 0.0

# run simulation
for ticks in range(5000):
    # update stimulus (this normally involves a call to the Stimulus object)
    if ticks == 1000:
        light_ON = True
        intensity = 100
        distance = 0.0
    elif ticks == 4000:
        light_ON = False
        intensity = 0
        distance = 0.0
    # update neurons
    for unit in range(len(E1_R2.neurons)):
        # present stimulus to neurons (this is normally handled by the Stimulus object)
        E1_R2.neurons[unit].stimulus_response(light_ON, intensity, distance)
        # trigger neurons to spike
        E1_R2.neurons[unit].update(light_ON, intensity, distance)

# analyze data
for unit in range(len(E1_R2.neurons)):
    print(E1_R2.neurons[unit], ':', len(E1_R2.neurons[unit].spike_train), E1_R2.neurons[unit].spike_train)


# In[ ]:

# initialize model
Neuron.id = 0
E1_R2 = Retina('wild_type')
num_units = 4
for unit in range(num_units):
    E1_R2.add_neuron(unit, neuron_type = 'RGC')

# run simulation with variable firing_rate
for ticks in range(10000):
    # update stimulus
    # present stimulus to neurons
    # trigger neurons to spike
    for unit in range(len(E1_R2.neurons)):
        E1_R2.neurons[unit].spiking()

# analyze data
for i in range(len(E1_R2.neurons)):
    unit = E1_R2.neurons[i]
    print(unit, ':', len(unit.spike_train), unit.spike_train)


# In[117]:

foo()


# In[116]:

print(E1_R2.neurons)

def foo():
    print('foo')
    return None


# In[82]:

Neuron.id = 0
channel = 1
retina = 'E1_R1'
cell1 = Neuron(channel, retina)
cell2 = Neuron(channel, retina)
print(cell1.id, cell2.id)


# In[118]:

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


# In[313]:

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









# In[78]:

# initialize model
E1_R2 = Retina('wild_type')
num_units = 4
for unit in range(num_units):
    E1_R2.add_neuron(unit)

# run simulation
for ticks in range(10000):
    for unit in range(len(E1_R2.neurons)):
        E1_R2.neurons[unit].spiking(10)

# analyze data
for i in range(len(E1_R2.neurons)):
    unit = E1_R2.neurons[i]
    print(len(unit.spike_train), unit.spike_train)


# In[160]:

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

mea = {'channel_map' : channel_map, 'offset_angle' : round(math.pi / 8, 2), 'offset_x' : 7,        'offset_y' : 3}
angle_list = np.linspace(0, 2 * np.pi, 12)
for i in range(len(angle_list)):
    angle_list[i] = round(angle_list[i], 2)
lifespan_list = []
speed = 100
for i in range(len(angle_list)):
    lifespan_list.append([])
    lifespan_list[i] = 1202.8 / speed
stimulus = {'angles' : angle_list, 'widths' : [10], 'lifespans' : lifespan_list, \
            'image_width' : 12000, 'image_height' : 9000, 'pixel_size' : 10.0, 'pinhole' : 1}

# read data from the stimulus dictionary
# read data from the stimulus dictionary
num_angles, angles, num_widths, widths, lifespans, image_width, image_height, pixel_size, pinhole = read_stimulus_dictionary(stimulus)

# read data from the mea dictionary
channel_map, channel_spacing, offset_angle, offset_x, offset_y = read_mea_dictionary(mea)

print('n_a:', num_angles, 'a:', angles, 'n_w:', num_widths, 'w:', widths, 'i_w:', 'l_s:', lifespans, \
      image_width, 'i_h:', image_height, 'p_sz:', pixel_size, 'p:', pinhole)
# read data from the mea dictionary

print('c_m:', channel_map)
print('c_s:', channel_spacing, 'o_a:', offset_angle, 'o_x:', offset_x, 'o_y:', offset_y)


# In[165]:

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
    num_angles, angles, image_width, image_height, pinhole = read_stimulus_dictionary(stimulus)

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

mea = {'channel_map' : channel_map, 'offset_angle' : math.pi / 8, 'offset_x' : 7,        'offset_y' : 3}
angle_list = np.linspace(0, 2 * np.pi, 12)
stimulus = {'angles' : angle_list, 'image_width' : 12000, 'pinhole' : 1}
channels = [1, 2, 3, 4]
DATh_dict = create_DATh_dictionary(stimulus, channels, mea)
print('DATh_dict:', DATh_dict)


# In[227]:

num_angles = 12
num_widths = 6
num_speeds = 6
num_intensities = 1
num_colors = 1
lifespans = []
for ang in range(num_angles):
    lifespans.append([])
    for wid in range(num_widths):
        lifespans[ang].append([])
        for spd in range(num_speeds):
            lifespans[ang][wid].append([])
            for it in range(num_intensities):
                lifespans[ang][wid][spd].append([])
                for col in range(num_colors):
                    lifespans[ang][wid][spd][it].append([wid+spd])



# In[228]:

print(lifespans)


# In[305]:

def find_next(search_list):
    print('search_list:', search_list)
    if type(search_list) is list:
        for item in search_list:
            print('inside item is:', item)
            if type(item) is list:
                next_list = next(find_next(item))
                print('next_list in if:', next_list)
                yield next_list
            else:
                print('else returning:', item)
                yield item
    else:
        yield search_item

a = 0
for item in lifespans:
    print('outside find_next, item is', item)
    print('result:', next(find_next(item)))
    a += 1
    print(a, 'times through')
    if a >= 14:
        break


# In[258]:

def find_next_lifespan(lifespans):
    item = 0
    while True:
        if type(lifespans[item]) is int:
            yield lifespans[item]
        else:
            find_next_lifespan(item)
        item += 1

print(next(find_first_lifespan(lifespans)))


# In[263]:

def next_lifespan(lifespans):
    """ lifespans is a list of nested lists
        Each element of is either #an int,# a float or a list
        No list is empty
        Generator that returns the *next* item in lifespans """

    def find_next(search_list):
        print('in find_next, type(lifespans) is', type(search_list))
        for item in search_list:
            if type(search_list) is list:
                search_list = find_next(item)
            else:
                return item

    count = 0
    next_item = lifespans
    while True:
        count += 1
        print(count, 'times through')
        if type(next_item) is list:
            print('in recursive call. next_item starts as:', next_item)
            for item in range(len(next_item)):
                next_item = find_next(next_item[item])
        else:
            print('yielding', next_item)
            yield next_item




t = [[1,2],[20,4, [8, [9, 17,[56, [23, 22, 21], [12, 2, [100]]],10]], 5, [6, 7]]]
a = 0
while True:
    print(a)
    next(next_lifespan(t))
    a += 1
    if a > 10:
        break


# In[265]:

def find_next(search_list):
    for item in search_list:
        if type(search_list) is list:
            find_next(item)
        else:
            yield item

t = [[1,2],[20,4, [8, [9, 17,[56, [23, 22, 21], [12, 2, [100]]],10]], 5, [6, 7]]]
a = 0
while True:
    print(a)
    next(find_next(t))
    a += 1
    if a > 10:
        break


# In[299]:

def find_next2(search_list):
    print('outside if, search_list is', search_list)
    if type(search_list) is list:
        for item in search_list:
            print('inside for loop, item is', item)
            result = next(find_next2(item))
            print('now result is', result)
            yield result
    else:
        print('inside else, returning', search_list)
        yield search_list
    print('after if, returning', result)
    yield result

t = [[1,2],[20,4, [8, [9, 17,[56, [23, 22, 21], [12, 2, [100]]],10]], 5, [6, 7]]]
a = 0
for item in t:
    a += 1
    print('find_next2',a,'is:',next(find_next2(item)))

    if a > 10:
        break


# In[ ]:
