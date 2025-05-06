'''
Helper functions to gather and process data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import (
    List, 
    Union, 
    Optional, 
    Dict, 
    Tuple
)
from seaborn import (
    lineplot, 
    swarmplot, 
    barplot, 
    histplot, 
    boxplot,
    violinplot,
    kdeplot,
    scatterplot,
    heatmap
)

from utils.measures import (
    PPT,
    OrderStrings, 
    ConditionalEntropy,
    PathUtils,
    GetMeasurements,
    Grid
)        
from utils.indices import AlternationIndex


class PlotStandardMeasures :
    '''
    Plots standard measures
    '''
    dpi = 300
    extension = 'png'
    width = 3
    height = 3.5
    cmaps = ["Blues", "Reds", "Greens", "Yellows"]
    regular_measures = [
        'attendance',
        'efficiency', 
        'inequality',
        'entropy',
        'conditional_entropy',
    ]
    standard_measures = [
        'attendance',
        'efficiency', 
        'inequality',
        'entropy',
        'conditional_entropy',
        'alternation_index'
    ]
    
    def __init__(self, data:pd.DataFrame) -> None:
        '''
        Input:
            - data, pandas dataframe
        '''
        self.data = data

    def plot_measures(
                self, 
                measures: List[str], 
                folder: Union[None, Path],
                kwargs: Optional[Union[Dict[str, str], None]]=None,
                categorical: Optional[bool]=False,
                suffix: Optional[Union[None, str]]=None
            ) -> List[Path]:
        # Tid up suffix
        if suffix is None:
            suffix = ''
        else:
            suffix = '_' + suffix
        # Tidy kwargs
        if kwargs is None:
            kwargs = dict()
        # Determine the number of rounds to plot
        T = kwargs.get('T', 20)
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        elif 'model_names' in kwargs.keys():
            assert(isinstance(kwargs['model_names'], dict))
            try:
                self.data.model = self.data.model.map(kwargs['model_names'])
            except:
                print("Warning: applying model names from kwargs didn't work.")
        try:
            self.data['model'] = self.data['model'].astype(int)
        except:
            try:
                self.data['model'] = self.data['model'].astype(float)
            except:
                pass
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        kwargs['num_models'] = num_models
        kwargs['vs_models'] = vs_models
        # Obtain data
        data = self.get_data(measures, T)
        # Initialize output list
        list_of_paths = list()
        # Plot per measure
        for m in measures:
            if folder is not None:
                file_ = PathUtils.add_file_name(folder, f'{m}{suffix}', self.extension)
            else:
                file_ = None
            print(f'Plotting {m}...')
            kwargs_ = kwargs.copy()
            if 'title' not in kwargs_.keys():
                kwargs_['title'] = m[0].upper() + m[1:]
            self.plot(
                measure=m, 
                data=data,
                kwargs=kwargs_,
                categorical=categorical,
                file=file_
            )
            if folder is not None:
                list_of_paths.append(file_)
        return list_of_paths	

    def plot(
                self, 
                measure: str,
                data: pd.DataFrame,
                kwargs: Dict[str,any],
                categorical: Optional[bool]=False,
                file: Optional[Union[Path, None]]=None
            ) -> Union[plt.axis, None]:
        '''
        Plots the variable against the models.
        Input:
            - kwargs: dict with additional setup values for plots
            - file, path of the file to save the plot on.
        Output:
            - None.
        '''
        assert(measure in self.standard_measures), f'Measure {measure} cannot be ploted by this class.'
        num_models = kwargs['num_models']
        vs_models = kwargs['vs_models']
        # Create the plot canvas
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (self.width * num_models, self.height)
        fig, ax = plt.subplots(
            figsize=figsize,
            tight_layout=True
        )
        variable = measure
        if 'with_treatment' in kwargs.keys() and kwargs['with_treatment']:
            if 'treatment' in data.columns:
                hue = 'treatment'
            else:
                hue = None
        else:
            hue = None
        if vs_models:
            if not categorical:
                lineplot(
                    x='model', y=variable, 
                    data=data, ax=ax, 
                    marker='o',
                    errorbar=('ci', 95)
                )
            else:
                boxplot(
                    x='model', y=variable, 
                    hue=hue,
                    data=data, ax=ax, 
                )
            ax.set_xlabel('Model')
            ax.set_ylabel(variable)
            # ax.set_ylim([-1.1, 1.1])
        else:
            histplot(
                data[variable],
                ax=ax
            )
            ax.set_xlabel(variable)
            # ax.set_xlim([-1.1, 1.1])
            ax.set_ylabel('Num. of episodes')
        # Set further information on plot
        ax = PlotStandardMeasures._customize_ax(ax, kwargs)
        # Save or return plot
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
            plt.close()
        else:
            print(f'Warning: No plot saved by plot_{measure}. To save plot, provide file name.')
            return ax

    @staticmethod
    def _customize_ax(
                ax:plt.axis, 
                kwargs:Dict[str,any]
            ) -> plt.axis:
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        return ax

    def plot_sweep2(
                self, 
                parameter1:str,
                parameter2:str,
                measure:str,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the average measure according to sweep of two parameters.
        Input:
            - parameter1, string with the first parameter name.
            - parameter2, string with the first parameter name.
            - measure, string with the measure.
            - T, integer with the length of the tail sequence.
            - file, string with the name of file to save the plot on.
            - kwargs, dictionary with extra tweaks for the plots.
        Output:
            - None.
        '''
        if T is None:
            T = 20
        annot = kwargs.get('annot', False)
        # If measure is alternation index, need to get more measures
        to_measure = AlternationIndex.complete_measures([measure])
        # Obtain data
        get_meas = GetMeasurements(
            self.data, measures=to_measure, T=T)
        get_meas.columns += [parameter2, parameter1]
        df = get_meas.get_measurements()
        if measure == 'alternation_index':
            index_gen = AlternationIndex.from_file()
            df['alternation_index'] = index_gen(df)
        df = df.groupby([parameter2, parameter1])[measure].mean().reset_index()
        values1 = df[parameter1].unique()
        values2 = df[parameter2].unique()
        df = pd.pivot(
            data=df,
            index=[parameter1],
            values=[measure],
            columns=[parameter2]
        ).reset_index().to_numpy()[:,1:]
        # Plotting...
        fig, ax = plt.subplots(figsize=(6,6))
        heatmap(data=df, ax=ax, annot=True)
        ax.set_xticklabels(np.round(values2, 2))
        ax.set_xlabel(parameter2)
        ax.set_yticklabels(np.round(values1, 2))
        ax.set_ylabel(parameter1)
        ax.set_title(f'Av. {measure}')
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        plt.close()

    def get_data(self, measures:List[str], T:int) -> pd.DataFrame:
        # Check if alternation index is in measures
        ai_dict = AlternationIndex.check_alternation_index_in_measures(measures)
        # Get other measures
        get_meas = GetMeasurements(
            self.data, 
            measures=ai_dict['measures'], 
            T=T
        )
        data = get_meas.get_measurements()
        ordered_models = OrderStrings.dict_as_numeric(data['model'].unique())
        data['model'] = data['model'].map(ordered_models)
        data.sort_values(by='model', inplace=True)
        # Add alternation index
        if ai_dict['check']:
            ai = AlternationIndex.from_file(priority='statsmodels')
            data['alternation_index'] = ai(data)
        return data


class PlotRoundMeasures(PlotStandardMeasures):
    '''
    Plot measures per round
    '''
    dpi = 300
    extension = 'png'
    width = 3
    height = 3.5
    cmaps = ["Blues", "Reds", "Greens", "Yellows"]
    round_measures = [
        'round_efficiency',
        'round_conditional_entropy'
    ]

    def __init__(self, data:pd.DataFrame) -> None:
        '''
        Input:
            - data, pandas dataframe
        '''
        super().__init__(data)

    def get_data(self, measures, T):
        get_meas = GetMeasurements(
            self.data, measures=measures, T=T, per_round=True
        )
        data = get_meas.get_measurements()
        ordered_models = OrderStrings.dict_as_numeric(data['model'].unique())
        data['model'] = data['model'].map(ordered_models)
        data.sort_values(by='model', inplace=True)
        return data

    def plot(
                self, 
                measure: str,
                data: pd.DataFrame,
                kwargs: Dict[str,any],
                categorical: Optional[bool]=False,
                file: Optional[Union[Path, None]]=None
            ) -> Union[plt.axis, None]:
        '''
        Plots the variable vs round per model.
        Input:
            - measure: str with the name of the measure to plot.
            - data: pandas dataframe with the data.
            - kwargs: dict with additional setup values for plots
            - file, path of the file to save the plot on.
        Output:
            - None or plt.axis.
        '''
        assert(measure in self.round_measures), f'Measure {measure} cannot be ploted by this class.'
        num_models = kwargs['num_models']
        vs_models = kwargs['vs_models']
        # Create the plot canvas
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (self.width * num_models, self.height)
        fig, ax = plt.subplots(
            figsize=figsize,
            tight_layout=True
        )
        variable = self.get_variable_from_measure(measure)
        if vs_models:
            lineplot(
                x='round', y=variable, 
                hue='model',
                data=data, ax=ax, 
                errorbar=('ci', 95)
            )
        else:
            lineplot(
                x='round', y=variable, 
                data=data, ax=ax, 
                errorbar=('ci', 95)
            )
        ax.set_xlabel('Round')
        ax.set_ylabel(variable)
        # Set further information on plot
        ax = PlotStandardMeasures._customize_ax(ax, kwargs)
        # Save or return plot
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
            plt.close()
        else:
            print('Warning: No plot saved by plot_efficiency. To save plot, provide file name.')
            return ax
        
    def get_variable_from_measure(self, measure:str) -> str:
        if measure == 'round_efficiency':
            return 'efficiency'
        if measure == 'round_conditional_entropy':
            return 'conditional_entropy'


class PlotVSMeasures:
    '''Plot 2D scatter plots'''
    dpi = 300
    extension = 'png'
    width = 3
    height = 3.5
    cmaps = ["Blues", "Reds", "Greens", "Yellows"]
    standard_measures = [
        'attendance',
        'efficiency', 
        'inequality',
        'entropy',
        'conditional_entropy',
    ]

    def __init__(self, data:pd.DataFrame) -> None:
        self.data = data
        self.warnings = True

    def two_way_comparisons(
                self, 
                measure_pairs:List[str],
                file:Path,
                kwargs:Optional[Dict[str,any]]={}
            ) -> Union[None, plt.axes]:
        vertical = kwargs.get('vertical', True)
        grid = Grid(len(measure_pairs), vertical)
        fig, axes = plt.subplots(
            grid.rows, grid.cols,
            figsize=(self.width*grid.cols, self.height*grid.rows)
        )
        kwargs_ = {
            'x_label':None,
            'y_label':None
        }
        kwargs.update(kwargs_)
        info_list = list()
        measures = list()
        for idx, ax in enumerate(axes):
            # Get pair of measures to plot
            pair_measures = measure_pairs[idx]
            # Add measures to list
            if pair_measures[0] not in measures:
                measures.append(pair_measures[0])
            if pair_measures[1] not in measures:
                measures.append(pair_measures[1])
            # Get plot and plot's info
            info = self.plot_vs(
                pair_measures=pair_measures,
                ax=ax,
                file=None, kwargs=kwargs
            )
            info_list.append(info)
        # Get max and min values for each measure
        dict_max_min = dict()
        for measure in measures:
            min_m = min([
                info[measure]['min'] for info in info_list if measure in info.keys()
            ])
            max_m = max([
                info[measure]['max'] for info in info_list if measure in info.keys()
            ])
            min_m = min_m - 0.1*(max_m - min_m)
            max_m = max_m + 0.1*(max_m - min_m)
            dict_max_min[measure] = [min_m, max_m]
        # Customize axes
        for idx, ax in enumerate(axes):        
            pair_measures = measure_pairs[idx]
            ax.set_xlabel(pair_measures[0])
            ax.set_ylabel(pair_measures[1])
            ax.set_xlim(dict_max_min[pair_measures[0]])
            ax.set_ylim(dict_max_min[pair_measures[1]])
            if idx == len(axes) - 1:
                # Get legend handles/labels from the last axes
                handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()
        # Place the legend below all subplots
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
        # Adjust spacing to accommodate legend
        fig.subplots_adjust(bottom=0.15)  # Add space for legend
        plt.tight_layout()
        # Save plot
        plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print('Plot saved to', file)

    def plot_vs(
                self,
                pair_measures:List[str],
                ax:Optional[Union[plt.axes, None]]=None,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> Union[None, plt.axes]:
        # Determine the number of rounds to plot
        T = kwargs.get('T', 20)
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        # Record measure names
        measure1, measure2 = pair_measures
        assert(measure1 in self.standard_measures)
        assert(measure2 in self.standard_measures)
        measures = [measure1, measure2]
        # Measure on the given measures
        gm = GetMeasurements(self.data, measures, T=T)
        df_measures = gm.get_measurements()
        # Jitter measures for better display
        sigma = 0.001
        df_measures[measure1] += np.random.normal(0,sigma, len(df_measures[measure1]))
        df_measures[measure2] += np.random.normal(0,sigma, len(df_measures[measure2]))
        # Save extremes
        info = dict()
        for measure in measures:
            info[measure] = dict()
            info[measure]['min'] = df_measures[measure].min()
            info[measure]['max'] = df_measures[measure].max()
        # Create the plot canvas
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlabel(f'{measure1[0].upper()}{measure1[1:]}')
        ax.set_ylabel(f'{measure2[0].upper()}{measure2[1:]}')
        # Plotting...
        if vs_models:
            # kdeplot(
            #     data=df_measures,
            #     x=measure1,
            #     y=measure2,
            #     hue='model',
            #     cmap=self.cmaps[:num_models], 
            #     fill=True,
            #     ax=ax,
            # )
            scatterplot(
                data=df_measures,
                x=measure1,
                y=measure2,
                hue='model',
                style='model',
                legend=True,
                ax=ax
            )
        else:
            # kdeplot(
            #     data=df_measures,
            #     x=measure1,
            #     y=measure2,
            #     cmap=self.cmaps[0],
            #     fill=True,
            #     ax=ax
            # )
            scatterplot(
                data=df_measures,
                x=measure1,
                y=measure2,
                ax=ax
            )
        # Set further information on plot
        ax = PlotStandardMeasures._customize_ax(ax, kwargs)
        # Save plot
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            print('Plot saved to', file)
        else:
            if self.warnings:
                print('Warning: No plot saved. To save plot, provide file name.')
            # fig.canvas.draw()
            # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to (H, W, 3)
            # return image
        return info

