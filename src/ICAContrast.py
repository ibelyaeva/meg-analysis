import mne
import numpy as np
import matplotlib as plt
from mne.viz.utils import _prepare_trellis
from mne.viz.utils import _setup_vmin_vmax
from mne.viz.utils import _setup_cmap
from mne.viz.utils import tight_layout
from mne.viz.utils import plt_show
import matplotlib.pyplot as plt
from mne.viz.topomap import _add_colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import save_plots_util as plt_util
import texfig
import math
plt.rcParams['text.usetex'] = True
plt.rcParams["legend.frameon"] = False
from mne import evoked as ev
from scipy import stats
from pdfCropMargins import crop


colors = {"AUD": "crimson", "VIS": 'steelblue'}

class ICA(object):
    def __init__(self, data,tc, evoked, times, component=None, contrast=None):
        self.data = data # (comp x subjects)
        self.info = evoked.info
        self.evoked = evoked
        
        self.tc = tc
        self.times = times
        self.n_comp = self.tc.shape[0]
        self.contrast = contrast
        self.component = component
            
        cnt = 0
        self._ica_names  = []
        for i in range(self.n_comp):
            cnt = cnt + 1
            self._ica_names.append('IC {:2d}'.format(cnt))
            
        cnt = 0   
        self._source_names = []
        for i in range(self.n_comp):
            cnt = cnt + 1
            self._source_names.append('IC{:2d}'.format(cnt))
       
     
    #plot all ICA Components    
    def plot_components(self,show=True, vmin=None, cmap = None, vmax=None, sensors=True, title=None, fig_id=None, colorbar=True):
        
        if title is None:
            title = 'Aggregate ERF Spatial Independent Components'
        
        scaling = 1e15
            
        fig, axes, _, _ = _prepare_trellis(len(self.data), ncols=self.data.shape[0])
        fig.suptitle(title)
        
        cmap = _setup_cmap(cmap, n_axes=1)
        
        titles = list()
        
        limit_vmin = []
        limit_vmax = []
        
        for data_ in self.data:
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
            limit_vmax.append(vmax_)
            limit_vmin.append(vmin_)
            
        vmin = np.min(np.array(limit_vmin))
        vmax = np.max(np.array(limit_vmax))
    
        print("Min = " + str(vmin))
        print("Max = " + str(vmax))
        
        cnt = 0
                 
        for data_, ax in zip(self.data, axes):
            kwargs = dict()
            titles.append(ax.set_title(self._ica_names[cnt], fontsize=10, **kwargs))
            
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
            
            im = mne.viz.plot_topomap(
            data_.flatten(), self.info, vmin=vmin_, vmax=vmax_, res=64, axes=ax,
            cmap=cmap[0],
            image_interp='spline16', show=False, sensors=sensors, ch_type='mag')[0]
            
            im.axes.set_label(self._ica_names[cnt])
            cnt = cnt + 1
            
        if colorbar:
            cbar, cax = _add_colorbar(ax, im, cmap, pad=.1, title="A.U",
                                  format='%3.2f')
            cbar.ax.tick_params(labelsize=10)
            cbar.set_ticks((vmin_, vmax_))
            
        tight_layout(fig=fig)
        fig.subplots_adjust(top=0.88, bottom=0.)
        
        if fig_id is not None:
            plt_util.save_fig_pdf(fig_id, tight_layout=True)
        
    def plot_ica_component(self, vmin=None, cmap = None, vmax=None, sensors=True, title=None, fig_id=None, colorbar=True):
        
        if title is None:
            title = 'Aggregate ERF Spatial'
            
        scaling = 1e15
        cmap = _setup_cmap(cmap, n_axes=1)
        cnt = 0
        
        limit_vmin = []
        limit_vmax = []
        
        for data_ in self.data:
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
            limit_vmax.append(vmax_)
            limit_vmin.append(vmin_)
            
        vmin = np.min(np.array(limit_vmin))
        vmax = np.max(np.array(limit_vmax))
    
        print("Min = " + str(vmin))
        print("Max = " + str(vmax))
        
          
        for data_ in self.data:
            fig, ax_topo = plt.subplots(1, 1, figsize=(3, 3))
            
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
            image = mne.viz.plot_topomap(
                    data_.flatten(), self.info, vmin=vmin_, vmax=vmax_, res=300, axes=ax_topo,
                    cmap=cmap[0],
                    image_interp='spline16', show=False, sensors=sensors, ch_type='mag')[0]
           
            image.axes.set_title(title + ' '  + self._ica_names[cnt], pad=0, fontsize=10)
            
            if colorbar:
                cbar, cax = _add_colorbar(ax_topo, image, cmap, pad=.1, title="A.U",
                                  format='%3.2f')
                cbar.ax.tick_params(labelsize=10)
                cbar.set_ticks((vmin_, vmax_))
              
            tight_layout(fig=fig)
            fig.subplots_adjust(top=0.95, bottom=0.01)
        
            if fig_id is not None:
                ica_fig_id = fig_id + '_ica_' + str(cnt+1)
                plt_util.save_fig_pdf_no_dpi(ica_fig_id)
                plt.close(fig)
            
            cnt = cnt + 1
                
        return fig
    
    def format_axes(self, ax):

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(1)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(1)

            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_tick_params(direction='in', top = True, left = True, right=True, color='k')
        
        return ax
    
    def plot_tc_grid(self, condition, fig_id = None,title=None, cols=3, figsize=(15,5), scaling=1e15):
        
        colors = ["crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                   "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson" ,"crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson","crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson"]
        
        nlines = self.tc.shape[0]
        nrows = math.ceil(nlines/cols)
        
        cmap = plt.cm.Spectral
        nlines = self.tc.shape[0]
        line_colors = cmap(np.linspace(0,1,nlines))
            
        times = self.times * 1e3
         
        x_label = "Time (ms)"
        y_label = "fT"
        
        if title is None:
            title = 'Aggregate ERF Temporal IC, Time-Locked. ' + r'$K=$' +str(nlines) 
              
        tmin = -100
        tmax = np.max( times)
             
        tc = self.tc
        
        tc = tc*scaling
        ymin = np.min(tc)
        ymax = np.max(tc)
        
        plt.vlines(4, 0, 5, linestyles ="dotted", colors ="k") 
        t_stim_end = 800
        
        fig, axes = plt.subplots(nrows=nrows, ncols=cols,  figsize=figsize)
                
        print(axes)
        
        times = self.times * 1e3
        x_ticks = np.arange(0,1050,50) 
        
        cnt = 0
        for i in range(nrows):
            for j in range(cols):
                
                if cnt >= nlines:
                    break
                
                try:
                    ax = axes[i, j]
                except:
                    ax = axes[cnt]
                self.format_axes(ax)
                          
                label = self._source_names[cnt]
                data = tc[cnt,:]
                data = data.flatten()
                print("data shape = " + str(data.shape))
                ax.plot(times, data, zorder=2, color=colors[cnt], linewidth=3, label=label)
                
                ax.set_xticks(x_ticks)
                ax.set_xlabel(x_label)
            
                ax.set_ylabel(y_label)
            
                ax.set_xlim(tmin, tmax)
                ax.set_ylim(ymin, ymax)
            
                ax.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
                ax.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
                            
                xy_coord = (0.83, 0.93)
                label_text = condition
                ax.annotate(
                    label_text, xy=xy_coord, xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    va='center', ha='left', color='black', size=6,fontsize=10)
            
                ax.legend(loc = 'upper right')
                
                cnt = cnt + 1
        
        #plt.subplots_adjust(top=0.07, bottom=0.04)
                
        if fig_id is not None:
            
            if title is None:
                title = 'Aggregate ERF Temporal IC, Time-Locked'
            
            fig.suptitle(title)
            
            file_fig_id = fig_id + '_evoked' + '_' + condition + '_' + str(cnt)
            tight_layout(fig=fig)
            plt_util.save_fig_pdf_no_dpi(file_fig_id)
            plt.close(fig) 
                
        return fig
            
    def plot_tc_grid_with_topo(self, condition, fig_id = None,title=None, cols=3, figsize=(15,5), scaling=1e15, topo_time=None, annotate=True):
        
        colors = ["crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                   "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson" ,"crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson","crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson"]
        
        nlines = self.tc.shape[0]
        nrows = math.ceil(nlines/cols)
        
        cmap = plt.cm.Spectral
        nlines = self.tc.shape[0]
        line_colors = cmap(np.linspace(0,1,nlines))
            
        times = self.times * 1e3
         
        x_label = "Time (ms)"
        y_label = "fT"
        
        if title is None:
            title = 'Aggregate ERF Temporal IC, Time-Locked. ' + r'$K=$' +str(nlines) 
              
        tmin = -100
        tmax = np.max( times)
             
        tc = self.tc
        
        tc = tc*scaling
        ymin = np.min(tc)
        ymax = np.max(tc)
        
        plt.vlines(4, 0, 5, linestyles ="dotted", colors ="k") 
        t_stim_end = 800
        
        fig, axes = plt.subplots(nrows=nrows, ncols=cols,  figsize=figsize)
                
        print(axes)
        
        times = self.times * 1e3
        x_ticks = np.arange(0,1050,100) 
        
        limit_vmin = []
        limit_vmax = []
        
        vmin=None
        vmax=None
        
        for data_ in tc:
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
            limit_vmax.append(vmax_)
            limit_vmin.append(vmin_)
            
        vmin = np.min(np.array(limit_vmin))
        vmax = np.max(np.array(limit_vmax))
    
        print("Min = " + str(vmin))
        print("Max = " + str(vmax))
        
        cnt = 0
        for i in range(nrows):
            for j in range(cols):
                
                if cnt >= nlines:
                    break
                
                try:
                    ax = axes[i, j]
                except:
                    ax = axes[cnt]
                self.format_axes(ax)
                          
                label = self._source_names[cnt]
                data = tc[cnt,:]
                data = data.flatten()
                print("data shape = " + str(data.shape))
                ax.plot(times, data, zorder=2, color=colors[cnt], linewidth=3, label=label)
                
                ax.set_xticks(x_ticks)
                ax.set_xlabel(x_label)
            
                ax.set_ylabel(y_label)
            
                ax.set_xlim(tmin, tmax)
                ax.set_ylim(ymin, ymax)
            
                ax.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
                ax.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
                            
                xy_coord = (0.8, 0.93)
                label_text = condition
                ax.annotate(
                    label_text, xy=xy_coord, xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    va='center', ha='left', color='black', size=6,fontsize=10)
            
                ax.legend(loc = 'upper right')
                
                cnt = cnt + 1
                
                vmin_, vmax_ = _setup_vmin_vmax(data, vmin, vmax)    
                ch_name, time_index = self.evoked.get_peak(ch_type='mag', mode='abs', tmin=0.0, tmax=0.400, time_as_index=True)
            
                time = self.evoked.times[time_index]
                print("peak time = " + str(time))
            
                comp_data = data.copy()
                max_loc400 = self.get_peak(comp_data, self.evoked.times, tmin=0.0, tmax=0.400, mode='abs')
                        
                comp_time400 = self.evoked.times[max_loc400]
                comp_time400vol = comp_data[max_loc400]
            
                print ("=component time 400=")
                print(max_loc400)
                print("comp_time 400=" + str(comp_time400))
                print("comp_time vol 400 =" + str(comp_time400vol))
                print ("=component time 400=")
            
            
                max_loc30 = self.get_peak(comp_data, self.evoked.times, tmin=0.0, tmax=0.030, mode='abs')
                comp_time30 = self.evoked.times[max_loc30]
                comp_time30vol = comp_data[max_loc30]
            
                print ("=component time 30=")
                print(max_loc30)
                print("comp_time 30 =" + str(comp_time30))
                print("comp_time vol 30=" + str(comp_time30vol))
                print ("=component time 30=")
            
            
                max_loc50 = self.get_peak(comp_data, self.evoked.times, tmin=0.06, tmax=0.065, mode='abs')
                comp_time50 = self.evoked.times[max_loc50]
                comp_time50vol = comp_data[max_loc50]
            
                print ("=component time 50=")
                print(max_loc50)
                print("comp_time 50=" + str(comp_time50))
                print("comp_time vol 50=" + str(comp_time50vol))
                print ("=component time 50=")
            
            
                max_loc100 = self.get_peak(comp_data, self.evoked.times, tmin=0.098, tmax=0.101, mode='abs')
                comp_time100 = self.evoked.times[max_loc100]
                comp_time100vol = comp_data[max_loc100]
            
                print ("=component time 100=")
                print(max_loc100)
                print("comp_time 100=" + str(comp_time100))
                print("comp_time vol 100=" + str(comp_time100vol))
                print ("=component time 100=")
                
                divider = make_axes_locatable(ax)
                ax_topo = divider.append_axes('right', size='20%', pad=0.01)
            
                ax_colorbar = divider.append_axes('right', size='2%', pad=0.05)
                        
                axes_list = []
                axes_list.append(ax_topo)
                axes_list.append(ax_colorbar)
            
                if topo_time is not None:
                    plot_topo_times = topo_time
                else:
                    plot_topo_times = comp_time400
            
                self.evoked.plot_topomap(times=plot_topo_times, ch_type='mag', 
                                     sensors=False, colorbar=True, scalings=1e15, 
                                     res=300, size=3,
                             time_unit='ms', contours=6, image_interp='spline16', average=0.02, axes=axes_list, extrapolate='head')
                
                #plot topomap
            
                #N300
                start = (comp_time400 - 0.01)*1e3
                stop =  (comp_time400 + 0.01)*1e3
                ymin, ymax = ax.get_ylim()
                print(ymin, ymax)
                print(start, stop)
                ax.fill_betweenx((ymin, ymax), start, stop,
                                   color='grey', alpha=0.2)
            
            
                #N25
                start = (comp_time30 - 0.005)*1e3
                stop =  (comp_time30 + 0.005)*1e3
                ymin, ymax = ax.get_ylim()
                print(ymin, ymax)
                print(start, stop)
                ax.fill_betweenx((ymin, ymax), start, stop,
                                   color='grey', alpha=0.2)
            
            
                #N50
                start = (comp_time50 - 0.005)*1e3
                stop =  (comp_time50 + 0.005)*1e3
                ymin, ymax = ax.get_ylim()
                print(ymin, ymax)
                print(start, stop)
                ax.fill_betweenx((ymin, ymax), start, stop,
                                    color='grey', alpha=0.2)
            
                #N100
                start = (comp_time100 - 0.005)*1e3
                stop =  (comp_time100 + 0.005)*1e3
                ymin, ymax = ax.get_ylim()
                print(ymin, ymax)
                print(start, stop)
                ax.fill_betweenx((ymin, ymax), start, stop,
                                   color='grey', alpha=0.2)
            
                if annotate:
               
                    xy_coord = (0.028, 0.05)
                    label_text = "P25m"
                    ax.annotate(
                        label_text, xy=xy_coord, xycoords='axes fraction',
                        xytext=(0, 0), textcoords='offset points',
                        va='center', ha='left', color='black', size=6,fontsize=8)
            
                    xy_coord = (0.1, 0.05)
                    label_text = "P50m"
                    ax.annotate(
                        label_text, xy=xy_coord, xycoords='axes fraction',
                        xytext=(0, 0), textcoords='offset points',
                        va='center', ha='left', color='black', size=6,fontsize=8)
            
                    xy_coord = (0.18, 0.05)
                    label_text = "N100m"
                    ax.annotate(
                        label_text, xy=xy_coord, xycoords='axes fraction',
                        xytext=(0, 0), textcoords='offset points',
                        va='center', ha='left', color='black', size=6,fontsize=8)
            
                    xy_coord = (0.4, 0.05)
                    label_text = "P300m"
                    ax.annotate(
                        label_text, xy=xy_coord, xycoords='axes fraction',
                        xytext=(0, 0), textcoords='offset points',
                        va='center', ha='left', color='black', size=6,fontsize=8)
        
        #plt.subplots_adjust(top=0.07, bottom=0.04)
                
        if fig_id is not None:
            
            if title is None:
                title = 'Aggregate ERF Temporal IC, Time-Locked'
            
            fig.suptitle(title)
            
            file_fig_id = fig_id + '_evoked' + '_' + condition + '_' + str(cnt)
            tight_layout(fig=fig)
            plt_util.save_fig_pdf_no_dpi(file_fig_id)
            plt.close(fig) 
                
        return fig    
    
    def plot_tc(self, condition, fig_id = None,title=None, scaling=1e15):
        #Plot average over epochs in ICA space.
        
             
        colors = ["crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                   "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson" ,"crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson","crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson"]
        
        cmap = plt.cm.Spectral
        nlines = self.tc.shape[0]
        line_colors = cmap(np.linspace(0,1,nlines))
            
        fig, axes = plt.subplots(1, figsize=(3, 3))
        ax = axes
        axes = [axes]
        times = self.times * 1e3
        
        lines = list()
        
        x_ticks = np.arange(0,1050,50)   
        ax.set(title=title, xlim=times[[0, -1]], xlabel='Time (ms)', ylabel='fT')
        
        cnt = 0
        
        x_label = "Time (ms)"
        y_label = "fT"
        
        if title is None:
            title = 'Aggregate ERF Temporal IC, Time-Locked'
              
        tmin = np.min( times)
        
        tmin = -100
        tmax = np.max( times)
        
             
        tc = self.tc*scaling
        ymin = np.min(tc)
        ymax = np.max(tc)
        
        plt.vlines(4, 0, 5, linestyles ="dotted", colors ="k") 
        t_stim_end = 800
        
    
        for data_ in tc:
            
            fig, axes = plt.subplots(1, figsize=(10, 3))
            
            self.format_axes(axes)
            
            label = self._source_names[cnt]
            print("data shape = " + str(data_.flatten().shape))
            axes.plot(times, data_.flatten(), zorder=2, color=colors[cnt], linewidth=3, label=label)
            axes.set_xticks(x_ticks)
            axes.set_xlabel(x_label)
            
            axes.set_ylabel(y_label)
            
            axes.set_xlim(tmin, tmax)
            axes.set_ylim(ymin, ymax)
            
            axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
            axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
            
            axes_title = title
            axes.set_title(axes_title)
            
            xy_coord = (0.85, 0.93)
            label_text = condition
            axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=10)
            
            axes.legend(loc = 'upper right')

            
            if fig_id is not None:
                file_fig_id = fig_id + '_evoked' + '_' + condition + '_' + str(cnt + 1)
                lines = [lines]
                tight_layout(fig=fig)
                plt_util.save_fig_pdf_no_dpi(file_fig_id)
                plt.close(fig)           
                
            cnt = cnt + 1
        
        return fig
    
    def create_info(self, data):
        
        ch_names = data.shape[0]
        info = mne.create_info(ch_names, 1000, ch_types='mag')
        add_lengh = data.shape[0] - len(self.info['ch_names'])
        
        print("Channels Name = " + str(self.info['ch_names']))
        
        cnt = 0
        channel_names = []
        for c in self.info['ch_names']:
            if (cnt < add_lengh):
                channel_names.append(c)
            cnt = cnt + 1
        
        info = mne.create_info(channel_names, 1000, ch_types='mag')
        info.set_montage(self.evoked.get_montage(), verbose=True)
        
        print("Montage = " + str(info))
        
        return info
    
    def get_peak(self,data, times, tmin=None, tmax=None, mode='abs'):
    
        if tmin is None:
            tmin = times[0]
        if tmax is None:
            tmax = times[-1]

        if tmin < times.min():
            raise ValueError('The tmin value is out of bounds. It must be '
                         'within {} and {}'.format(times.min(), times.max()))
        if tmax > times.max():
            raise ValueError('The tmax value is out of bounds. It must be '
                         'within {} and {}'.format(times.min(), times.max()))
        if tmin > tmax:
            raise ValueError('The tmin must be smaller or equal to tmax')

        time_win = (times >= tmin) & (times <= tmax)
        mask = np.ones_like(data).astype(np.bool)
        mask[time_win] = False

        maxfun = np.argmax
        if mode == 'pos':
            if not np.any(data > 0):
                raise ValueError('No positive values encountered. Cannot '
                             'operate in pos mode.')
        elif mode == 'neg':
            if not np.any(data < 0):
                raise ValueError('No negative values encountered. Cannot '
                             'operate in neg mode.')
            maxfun = np.argmin

        masked_index = np.ma.array(np.abs(data) if mode == 'abs' else data,
                               mask=mask)

        max_loc = np.unravel_index(maxfun(masked_index), data.shape)

        return max_loc
    
    def plot_tc_with_topo(self, condition, fig_id = None,title=None, scalings=1e15, annotate=True, topo_time=None):
        #Plot average over epochs in ICA space.
        
             
        colors = ["crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                   "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson" ,"crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson","crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson"]
        
        cmap_ = plt.cm.Spectral
        nlines = self.tc.shape[0]
        line_colors = cmap_(np.linspace(0,1,nlines))
            
        fig, axes = plt.subplots(1, figsize=(3, 3))
        ax = axes
        axes = [axes]
        times = self.times * 1e3
        
        lines = list()
        
        x_ticks = np.arange(0,1050,50)   
        ax.set(title=title, xlim=times[[0, -1]], xlabel='Time (ms)', ylabel='fT')
        
        cnt = 0
        
        x_label = "Time (ms)"
        y_label = "fT"
        
        if title is None:
            title = 'Aggregate ERF Temporal IC, Time-Locked'
        scaling = 1e15
        
        tmin = np.min( times)
        
        tmin = -100
        tmax = np.max( times)
        
             
        tc = self.tc*scalings
        ymin = np.min(tc)
        ymax = np.max(tc)
        
        plt.vlines(4, 0, 5, linestyles ="dotted", colors ="k") 
        t_stim_end = 800
        
        cmap=None
        cmap = _setup_cmap(cmap, n_axes=1)
        cnt = 0
        
        limit_vmin = []
        limit_vmax = []
        
        vmin=None
        vmax=None
        
        for data_ in tc:
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
            limit_vmax.append(vmax_)
            limit_vmin.append(vmin_)
            
        vmin = np.min(np.array(limit_vmin))
        vmax = np.max(np.array(limit_vmax))
    
        print("Min = " + str(vmin))
        print("Max = " + str(vmax))
        
        for data_ in tc:
            
            fig, axes = plt.subplots(1, figsize=(10, 3))
            
            self.format_axes(axes)
            
            label = self._source_names[cnt]
            print("data shape = " + str(data_.flatten().shape))
            axes.plot(times, data_.flatten(), color=colors[cnt], linewidth=3, label=label)
            axes.set_xticks(x_ticks)
            axes.set_xlabel(x_label)
            
            axes.set_ylabel(y_label)
            
            axes.set_xlim(tmin, tmax)
            axes.set_ylim(ymin, ymax)
            
            axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
            axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
            
            axes_title = title
            axes.set_title(axes_title)
            
            xy_coord = (0.85, 0.93)
            label_text = condition
            axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=10)
            
            axes.legend(loc = 'upper right')
            
                      
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)    
            ch_name, time_index = self.evoked.get_peak(ch_type='mag', mode='abs', tmin=0.0, tmax=0.400, time_as_index=True)
            
            time = self.evoked.times[time_index]
            print("peak time = " + str(time))
            
            comp_data = data_.copy()
            max_loc400 = self.get_peak(comp_data, self.evoked.times, tmin=0.0, tmax=0.400, mode='abs')
                        
            comp_time400 = self.evoked.times[max_loc400]
            comp_time400vol = comp_data[max_loc400]
            
            print ("=component time 400=")
            print(max_loc400)
            print("comp_time 400=" + str(comp_time400))
            print("comp_time vol 400 =" + str(comp_time400vol))
            print ("=component time 400=")
            
            
            max_loc30 = self.get_peak(comp_data, self.evoked.times, tmin=0.0, tmax=0.030, mode='abs')
            comp_time30 = self.evoked.times[max_loc30]
            comp_time30vol = comp_data[max_loc30]
            
            print ("=component time 30=")
            print(max_loc30)
            print("comp_time 30 =" + str(comp_time30))
            print("comp_time vol 30=" + str(comp_time30vol))
            print ("=component time 30=")
            
            
            max_loc50 = self.get_peak(comp_data, self.evoked.times, tmin=0.06, tmax=0.065, mode='abs')
            comp_time50 = self.evoked.times[max_loc50]
            comp_time50vol = comp_data[max_loc50]
            
            print ("=component time 50=")
            print(max_loc50)
            print("comp_time 50=" + str(comp_time50))
            print("comp_time vol 50=" + str(comp_time50vol))
            print ("=component time 50=")
            
            
            max_loc100 = self.get_peak(comp_data, self.evoked.times, tmin=0.098, tmax=0.101, mode='abs')
            comp_time100 = self.evoked.times[max_loc100]
            comp_time100vol = comp_data[max_loc100]
            
            print ("=component time 100=")
            print(max_loc100)
            print("comp_time 100=" + str(comp_time100))
            print("comp_time vol 100=" + str(comp_time100vol))
            print ("=component time 100=")
            
            #plot topomap
            divider = make_axes_locatable(axes)
            ax_topo = divider.append_axes('right', size='20%', pad=0.01)
            
            ax_colorbar = divider.append_axes('right', size='2%', pad=0.05)
                        
            axes_list = []
            axes_list.append(ax_topo)
            axes_list.append(ax_colorbar)
            
            if topo_time is not None:
                plot_topo_times = topo_time
            else:
                plot_topo_times = comp_time400
            
            self.evoked.plot_topomap(times=plot_topo_times, ch_type='mag', 
                                     sensors=False, colorbar=True, scalings=1e15, 
                                     res=300, size=3,
                             time_unit='ms', contours=6, image_interp='spline16', average=0.02, axes=axes_list, extrapolate='head')
            
            #plot topomap
            
            #N300
            start = (comp_time400 - 0.01)*1e3
            stop =  (comp_time400 + 0.01)*1e3
            ymin, ymax = axes.get_ylim()
            print(ymin, ymax)
            print(start, stop)
            axes.fill_betweenx((ymin, ymax), start, stop,
                                   color='grey', alpha=0.2)
            
            
            #N25
            start = (comp_time30 - 0.005)*1e3
            stop =  (comp_time30 + 0.005)*1e3
            ymin, ymax = axes.get_ylim()
            print(ymin, ymax)
            print(start, stop)
            axes.fill_betweenx((ymin, ymax), start, stop,
                                   color='grey', alpha=0.2)
            
            
            #N50
            start = (comp_time50 - 0.005)*1e3
            stop =  (comp_time50 + 0.005)*1e3
            ymin, ymax = axes.get_ylim()
            print(ymin, ymax)
            print(start, stop)
            axes.fill_betweenx((ymin, ymax), start, stop,
                                    color='grey', alpha=0.2)
            
            #N100
            start = (comp_time100 - 0.005)*1e3
            stop =  (comp_time100 + 0.005)*1e3
            ymin, ymax = axes.get_ylim()
            print(ymin, ymax)
            print(start, stop)
            axes.fill_betweenx((ymin, ymax), start, stop,
                                   color='grey', alpha=0.2)
            
            if annotate:
               
                xy_coord = (0.035, 0.05)
                label_text = "P25m"
                axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=8)
            
                xy_coord = (0.1, 0.05)
                label_text = "P50m"
                axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=8)
            
                xy_coord = (0.18, 0.05)
                label_text = "N100m"
                axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=8)
            
                xy_coord = (0.4, 0.05)
                label_text = "P300m"
                axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=8)
            
            if fig_id is not None:
                file_fig_id = fig_id + '_evoked' + '_' + condition + '_' + str(cnt + 1)
                lines = [lines]
                tight_layout(fig=fig)
                plt_util.save_fig_pdf_no_dpi(file_fig_id)
                plt.close(fig)           
                
            cnt = cnt + 1
        
        return fig
    
    def plot_tc_contrast_with_topo(self, condition, col_num, fig_id = None,title=None, scalings=1e15, annotate=True, topo_time=None):
        #Plot average over epochs in ICA space.
        
             
        colors = ["crimson", "#DAA520", "steelblue", "crimson", "crimson", "crimson",
                   "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson" ,"crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson","crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson",
                    "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson"]
        
        
        label_font = {
        'size': 9,
        }
        
        cmap_ = plt.cm.Spectral
        nlines = self.tc.shape[0]
        line_colors = cmap_(np.linspace(0,1,nlines))
            
        times = self.times * 1e3
        x_ticks = np.arange(0,1100,100)   
            
        x_label = "Time (ms)"
        y_label = "fT"
        
        if title is not None:
            title = title
       
        tmin = -100
        tmax = np.max( times)
                 
        tc = self.tc*scalings
            
        cmap=None
        cmap = _setup_cmap(cmap, n_axes=1)
    
        limit_vmin = []
        limit_vmax = []
        
        vmin=None
        vmax=None
        
        for data_ in tc:
            vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
            limit_vmax.append(vmax_)
            limit_vmin.append(vmin_)
            
        vmin = np.min(np.array(limit_vmin))
        vmax = np.max(np.array(limit_vmax))
    
        print("Min = " + str(vmin))
        print("Max = " + str(vmax))
        
        fig, axes = plt.subplots(1, figsize=(7, 3))
        self.format_axes(axes)
        t_stim_end = 800
        
        component = self.component['component']
        component_data = component.data*scalings
        component_label = component.contrast_name
        
        component_fist = self.component['first']
        component_fist_data = component_fist.data[0]*1e3
        component_fist_label = component_fist.contrast_name
        
        component_second = self.component['second']
        component_second_data = component_second.data[0]*1e3
        component_second_label = component_second.contrast_name
               
        print("component data shape = " + str(component_data.flatten().shape))
        print("component name = " + str(component_label))
        
        print("fist group data shape = " + str(component_fist_data.flatten().shape))
        print("component 1st name = " + str(component_fist_label))
        
        print("second group data shape = " + str(component_second_data.flatten().shape))
        print("component 1st name = " + str(component_second_label))
        
        ymin1 = np.min(component_data)
        ymin2 = np.min(component_fist_data)
        ymin3 = np.min(component_second_data)
        ymin_arr = np.asarray([ymin1, ymin2, ymin3])
        ymin = np.amin(ymin_arr)

        ymax1 = np.max(component_data)
        ymax2 = np.max(component_fist_data)
        ymax3 = np.max(component_second_data)
        ymax_arr = np.asarray([ymax1, ymax2, ymax3])
        
        ymax = np.amax(ymax_arr)
        
        
        cnt = 0
        if component.p_val <= 0.05:
            component_label = component.contrast_name +  r'$^*$'
        else:
            component_label = component.contrast_name
            
        axes.plot(times, component_data.flatten(), color=colors[cnt], linewidth=3, label=component_label, zorder=10)
        axes.set_xticks(x_ticks)
        axes.set_xticks(x_ticks)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_xlim(tmin, tmax)
        axes.set_ylim(ymin, ymax)
            
        axes_title = title
        axes.set_title(axes_title)
            
        cnt = cnt + 1
        
        axes.plot(times, component_fist_data.flatten(), color=colors[cnt], linewidth=3, label=component_fist_label, zorder=9)
        cnt = cnt + 1
        
        axes.plot(times, component_second_data.flatten(), color=colors[cnt], linewidth=3, label=component_second_label, zorder=8)
        
        bbox_to_anchor=(0.3, -0.1)
        #axes.legend(loc=(0.6, 0.08))
        axes.legend(loc=(0.0,-0.4), ncol=3)
        
        axes.vlines(0, ymin, ymax, linestyles ="dotted", linewidth=0.5, colors ="k")
        axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", linewidth=0.5, colors ="k")
        
        
        component_fist_data_cp= component_fist_data.copy().flatten()
        print(component_fist_data_cp.shape)
        max_loc400 = self.get_peak(component_fist_data_cp, self.evoked.times, tmin=0.2, tmax=0.400, mode='abs')
                        
        comp_time400 = self.evoked.times[max_loc400]
        comp_time400vol = component_fist_data_cp[max_loc400]
        
        start = (comp_time400 - 0.01)*1e3
        stop =  (comp_time400 + 0.01)*1e3
        ymin, ymax = axes.get_ylim()
        print(ymin, ymax)
        print(start, stop)
        #axes.fill_betweenx((ymin, ymax), start, stop,
        #                           color='#DAA520', alpha=0.1) //dont fill for the HIGH IC component here
        
        #plot first group IC
        #ax_group1 = divider.append_axes('right', size='20%', pad=0.01)
    
        
        #plot first group IC
        divider = make_axes_locatable(axes)
        ax_group1_ic = divider.append_axes('right', size='20%', pad=0.02)
        
        if topo_time is not None:
                plot_topo_times = topo_time
        else:
                plot_topo_times = comp_time400
          
           
        group1_ic = component_fist.evoked.plot_topomap(times=plot_topo_times, ch_type='mag', 
                                     sensors=False, colorbar=False, scalings=1e3, 
                                     res=300, size=3,
                             time_unit='ms', contours=6, image_interp='spline16', average=0.02, axes=ax_group1_ic, extrapolate='head')
        
        ax_group1_ic.set_xlabel(component_fist.contrast_name, fontdict=label_font)
        
        component_second_data_cp= component_second_data.copy().flatten()
        print(component_second_data_cp.shape)
        max_loc400 = self.get_peak(component_second_data_cp, self.evoked.times, tmin=0.2, tmax=0.400, mode='abs')
                        
        comp_time400 = self.evoked.times[max_loc400]
        comp_time400vol = component_second_data_cp[max_loc400]
        
        start = (comp_time400 - 0.01)*1e3
        stop =  (comp_time400 + 0.01)*1e3
        ymin, ymax = axes.get_ylim()
        print(ymin, ymax)
        print(start, stop)
        axes.fill_betweenx((ymin, ymax), start, stop,
                                  color='steelblue', alpha=0.1)
        
        #plot second group IC
        ax_group2_ic = divider.append_axes('right', size='20%', pad=0.03)
        
        if topo_time is not None:
                plot_topo_times = topo_time
        else:
                plot_topo_times = comp_time400
          
           
        group2_ic = component_second.evoked.plot_topomap(times=plot_topo_times, ch_type='mag', 
                                     sensors=False, colorbar=False, scalings=1e3, 
                                     res=300, size=3,
                             time_unit='ms', contours=6, image_interp='spline16', average=0.02, axes=ax_group2_ic, extrapolate='head')
        
        ax_group2_ic.set_xlabel(component_second.contrast_name, fontdict=label_font)
        
        
        comp_data = component_data.copy()
        max_loc400 = self.get_peak(comp_data, self.evoked.times, tmin=0.0, tmax=0.400, mode='abs')
                        
        comp_time400 = self.evoked.times[max_loc400]
        comp_time400vol = comp_data[max_loc400]
            
        print ("=component time 400=")
        print(max_loc400)
        print("comp_time 400=" + str(comp_time400))
        print("comp_time vol 400 =" + str(comp_time400vol))
        print ("=component time 400=")
        
        start = (comp_time400 - 0.01)*1e3
        stop =  (comp_time400 + 0.01)*1e3
        ymin, ymax = axes.get_ylim()
        print(ymin, ymax)
        print(start, stop)
        axes.fill_betweenx((ymin, ymax), start, stop,
                                   color='crimson', alpha=0.1)
        
        #plot topomap IC Component
        ax_topo = divider.append_axes('right', size='20%', pad=0.03)
            
        ax_colorbar = divider.append_axes('right', size='2%', pad=0.05)
                        
        axes_list = []
        axes_list.append(ax_topo)
        axes_list.append(ax_colorbar)
            
        if topo_time is not None:
                plot_topo_times = topo_time
        else:
                plot_topo_times = comp_time400
          
           
        vef_ic = component.evoked.plot_topomap(times=plot_topo_times, ch_type='mag', 
                                     sensors=False, colorbar=True, scalings=1e15, 
                                     res=300, size=3,
                             time_unit='ms', contours=6, image_interp='spline16', average=0.02, axes=axes_list, extrapolate='head')
        
        if component.p_val <= 0.05:
            component_label = component.contrast_name +  r'$^*$' + '\n' + r'$p={:.5f}$'.format(component.p_val)
        else:
            component_label = component.contrast_name + '\n' + r'$p={:.5f}$'.format(component.p_val)
            
        ax_topo.set_xlabel(component_label, fontdict=label_font)
        #plot topomap
        print("component_label.p_val = " + str(component.p_val))
        
        
        max_loc50 = self.get_peak(comp_data, self.evoked.times, tmin=0.06, tmax=0.065, mode='abs')
        comp_time50 = self.evoked.times[max_loc50]
        comp_time50vol = comp_data[max_loc50]
            
        print ("=component time 50=")
        print(max_loc50)
        print("comp_time 50=" + str(comp_time50))
        print("comp_time vol 50=" + str(comp_time50vol))
        print ("=component time 50=")
            
            
        max_loc100 = self.get_peak(comp_data, self.evoked.times, tmin=0.098, tmax=0.101, mode='abs')
        comp_time100 = self.evoked.times[max_loc100]
        comp_time100vol = comp_data[max_loc100]
            
        print ("=component time 100=")
        print(max_loc100)
        print("comp_time 100=" + str(comp_time100))
        print("comp_time vol 100=" + str(comp_time100vol))
        print ("=component time 100=")
        
        #N50
        start = (comp_time50 - 0.005)*1e3
        stop =  (comp_time50 + 0.005)*1e3
        ymin, ymax = axes.get_ylim()
        print(ymin, ymax)
        print(start, stop)
        axes.fill_betweenx((ymin, ymax), start, stop,
                                    color='grey', alpha=0.1)
            
        #N100
        start = (comp_time100 - 0.005)*1e3
        stop =  (comp_time100 + 0.005)*1e3
        ymin, ymax = axes.get_ylim()
        print(ymin, ymax)
        print(start, stop)
        axes.fill_betweenx((ymin, ymax), start, stop,
                                   color='grey', alpha=0.1)
    
            
        
        
        if annotate:
            
                xy_coord = (0.07, 0.05)
                label_text = "P50m"
                axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=8)
            
                xy_coord = (0.18, 0.05)
                label_text = "N100m"
                axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=8)
            
                xy_coord = (0.4, 0.05)
                label_text = "P300m"
                axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=8)
        
        
        plt.subplots_adjust(bottom=0.01)
        
        if fig_id is not None:
                file_fig_id = fig_id + '_component' + '_' + condition + '_' + str(col_num)
                tight_layout(fig=fig)
                plt_util.save_fig_pdf_no_dpi(file_fig_id)
                #crop(["-p", "0", "-v", "-u", "-s", file_fig_id + ".pdf"])
                plt.close(fig)

        
       
        return fig
    
    
        
            
