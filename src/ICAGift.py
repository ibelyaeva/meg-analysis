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
plt.rcParams['text.usetex'] = True
plt.rcParams["legend.frameon"] = False


colors = {"AUD": "crimson", "VIS": 'steelblue'}

class ICA(object):
    def __init__(self, data, info, tc, times):
        self.data = data # (comp x subjects)
        self.info = info
        self.evoked = mne.EvokedArray(self.data.T, self.info, tmin=0,
                                      nave=self.data.T.shape[0], verbose=True)
        self.n_comp = self.data.shape[0]
    
        self.tc = tc
        self.times = times
        
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
    
    def plot_tc(self, condition, fig_id = None,title=None, labels=None):
        #Plot average over epochs in ICA space.
        
        colors = {'AUD': "crimson"}
        
        colors = ["crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson", "crimson"]
            
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
        
        self.tc = self.tc*scaling
        
        ymin = np.min(self.tc)
        ymax = np.max(self.tc)
        
        plt.vlines(4, 0, 5, linestyles ="dotted", colors ="k") 
        t_stim_end = 800
        
        for data_ in self.tc:
            
            fig, axes = plt.subplots(1, figsize=(14, 3))
            
            self.format_axes(axes)
            
            label = self._source_names[cnt] + r'$_{T}$'
            print("data shape = " + str(data_.flatten().shape))
            axes.plot(times, data_.flatten(), zorder=2, color=colors[cnt], label=label)
            axes.set_xticks(x_ticks)
            axes.set_xlabel(x_label)
            
            axes.set_ylabel(y_label)
            
            axes.set_xlim(tmin, tmax)
            axes.set_ylim(ymin, ymax)
            
            axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
            axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
            
            axes_title = title
            axes.set_title(axes_title)
            
            xy_coord = (0.9, 0.93)
            label_text = condition
            axes.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=10)
            
            axes.legend()

            
            if fig_id is not None:
                file_fig_id = fig_id + '_evoked' + '_' + condition + '_' + str(cnt + 1)
                lines = [lines]
                tight_layout(fig=fig)
                plt_util.save_fig_pdf_no_dpi(file_fig_id)
                plt.close(fig)           
                
            cnt = cnt + 1
        
        return fig
    
    def plot_tc1(self, condition, fig_id = None):
        
        if condition == 'AUD':
            colors = {'AUD': "crimson"}
            linestyles = {'AUD': '-'}
            styles = {"AUD": {"linewidth": 4}}
        else:
            colors = {"VIS": "steelblue"}
            linestyles = {"VIS": '-'}
            styles = {"VIS": {"linewidth": 4}}
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            
        image = mne.viz.plot_compare_evokeds(self.tc, title='Temporal IC', axes=ax,
                         colors=colors, show=True, linestyles=linestyles,  ci=True,
                         split_legend=True, truncate_yaxis='auto')
        
        
        print ("I AM HERE")
        
        print("Image")
        print(image)
        
        if fig_id is not None:
            file_fig_id = fig_id + '_evoked'
            plt_util.save_fig_pdf_no_dpi(file_fig_id)
            plt.close(fig)           
        
        plt.show()
            
           
        
            
