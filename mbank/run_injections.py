from mbank.metric import variable_handler, cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import load_PSD, plot_tiles_templates, get_boundaries_from_ranges, plot_match_histogram, plot_colormap
from mbank.placement import place_random_flow
from mbank.flow import STD_GW_Flow
from mbank.flow.utils import early_stopper, plot_loss_functions
from tqdm import tqdm
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from mbank.utils import compute_injections_match, ray_compute_injections_match, get_random_sky_loc, initialize_inj_stat_dict, save_inj_stat_dict

variable_format = 'mcqlambdatilde_chi'
psdfile = 'aligo_O3actual_H1.txt'
bank = cbc_bank(variable_format)
bank.load('bank_chi.dat')

flow = STD_GW_Flow(4, n_layers=4, hidden_features=50)
flow.load_weights('flow_chi.zip')

metric = cbc_metric(variable_format,
			PSD = load_PSD(psdfile, True, 'H1'),
			approx = 'IMRPhenomD_NRTidal',
			f_min = 10, f_max = 1024)

n_injs = 16000
# injs_3D = flow.sample(n_injs)
injs_3D = flow.sample(n_injs).detach().numpy()

injs_12D = bank.var_handler.get_BBH_components(injs_3D, bank.variable_format)
sky_locs = np.column_stack(get_random_sky_loc(n_injs))
stat_dict = initialize_inj_stat_dict(injs_12D, sky_locs = sky_locs)

	##
	# Computing the injection match
	##
inj_stat_dict = ray_compute_injections_match(stat_dict, bank,
	metric_obj = metric, mchirp_window = 0.1, symphony_match = True, max_jobs=8)
save_inj_stat_dict('injections_chi_plus.json', inj_stat_dict)

matches = inj_stat_dict['match']  # True matches from waveform overlaps
matches_metric = inj_stat_dict.get('metric_match', None)
	##
	# Plotting
	##
plot_tiles_templates(bank.templates, bank.variable_format,
	injections = injs_3D, inj_cmap = stat_dict['match'], show = True)

plt.savefig('injection_match_chi_plus.png')

threshold = 0.9
fraction_above = np.mean(matches >= threshold)
percent_above = 100 * fraction_above

print(f"{percent_above:.1f}% of injections have matches >= {threshold}")

plot_match_histogram(
    matches_metric=matches_metric,
    matches=matches,
    mm=0.97,
    bank_name="BNS Tidal Bank"
)

plt.savefig('match_histogram_chi_plus.png', dpi=150, bbox_inches='tight')
plt.close()

plot_colormap(
    datapoints=injs_3D,
    values=matches,
    variable_format=variable_format,
    statistics="mean",
    bins=40,
    fs=14,
    values_label="Match",
    savefile="colormap_mc_lambdatilde_match.png",
    show=True,
    title="Mean Match"
)
