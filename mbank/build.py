"""
Referred from "Example on how to generate a template bank by hand: https://github.com/stefanoschmidt1995/mbank/blob/master/examples/bank_by_hand.py". Generating a bank with tides.
"""
	##
	# Imports
	##
from mbank.metric import variable_handler, cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import load_PSD, plot_tiles_templates, get_boundaries_from_ranges
from mbank.placement import place_random_flow
from mbank.flow import STD_GW_Flow
from mbank.flow.utils import early_stopper, plot_loss_functions
from tqdm import tqdm
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

	##
	# Initializing the metric & boundaries
	##

variable_format = 'mcqlambdatilde_chi'
dim = 4

boundaries = get_boundaries_from_ranges(variable_format,
		M_range=(1.0, 1.7),      # chirp mass
    	q_range=(1.0, 3.0),
		chi_range=(-0.05, 0.05),
		lambdatilde_range=(100.0, 3000.0),)

print(f"Boundaries shape: {boundaries.shape}")
print(f"Boundaries:\n{boundaries}")

psdfile = 'aligo_O3actual_H1.txt'
metric = cbc_metric(variable_format,
			PSD = load_PSD(psdfile, True, 'H1'),
			approx = 'IMRPhenomD_NRTidal',
			f_min = 10, f_max = 1024)

	##
	# Generating training data
	##

train_data = np.random.uniform(*boundaries, (15000, 4))
validation_data = np.random.uniform(*boundaries, (500, 4))
train_ll = np.array([metric.log_pdf(s) for s in tqdm(train_data)])
validation_ll = np.array([metric.log_pdf(s) for s in tqdm(validation_data)])

	##
	# Initializing, training and testing the normalizing flow
	##

flow = STD_GW_Flow(4, n_layers = 4, hidden_features = 50)

early_stopper_callback = early_stopper(patience=20, min_delta=1e-3)
optimizer = optim.Adam(flow.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = .02, factor = 0.5, patience = 4)
	
history = flow.train_flow('ll_mse', N_epochs = 15000,
	train_data = train_data, train_weights = train_ll,
	validation_data = validation_data, validation_weights = validation_ll,
	optimizer = optimizer, batch_size = 500, validation_step = 100,
	callback = early_stopper_callback, lr_scheduler = scheduler,
	boundaries = boundaries, verbose = True)

residuals = np.squeeze(validation_ll) - flow.log_volume_element(validation_data)

	##
	# Placing the templates
	##

new_templates = place_random_flow(0.97, flow, metric,
	n_livepoints = 1000, covering_fraction = 0.95,
	boundaries_checker = boundaries,
	metric_type = 'symphony', verbose = True)
bank = cbc_bank(variable_format)
bank.add_templates(new_templates)

	##
	# Saving the template banks and the flow
	##

flow.save_weigths('flow_chi.zip')
bank.save_bank('bank_chi.dat')

	##
	# Doing some plots
	##




vh = variable_handler()
Mchirp = vh.get_mchirp(bank.templates, variable_format)
q = vh.get_massratio(bank.templates, variable_format)
lambdatilde = bank.templates[:, 2]
chi = bank.templates[:, 3]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(Mchirp, q, s=5)
axes[0].set_xlabel('Mchirp [M☉]')
axes[0].set_ylabel('q')

axes[1].scatter(Mchirp, lambdatilde, s=5)
axes[1].set_xlabel('Mchirp ')
axes[1].set_ylabel('lambda_tilde')

axes[2].scatter(chi, lambdatilde, s=5)
axes[2].set_xlabel('s1z')
axes[2].set_ylabel('lambda_tilde')

plt.tight_layout()
plt.savefig('bank_plots_chi.png', dpi=300)
plt.show()

