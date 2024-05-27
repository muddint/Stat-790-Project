import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datetime import datetime
from findiff_helpers import MLPSynthesizer, BaseDiffuser


class FinDiffSynthesizer:
    def __init__(self, data: pd.DataFrame, categorical_columns):
        data = data.copy()
        # set dimension of categorical embeddings
        self.cat_emb_dim = 2
        # set number of neurons per layer
        self.mlp_layers = [1024, 1024, 1024, 1024]
        # set non-linear activation function
        activation = 'lrelu'
        # set number of diffusion steps
        diffusion_steps = 500
        # set diffusion start and end betas
        diffusion_beta_start = 1e-4
        diffusion_beta_end = 0.02
        # set diffusion scheduler
        scheduler = 'linear'
        # set number of training epochs
        self.epochs = 30
        # set training batch size
        batch_size = 512
        # set training learning rate
        learning_rate = 1e-4
        # set seed
        self.seed = 42

        # original column names
        self.original_column_names = {col.replace('_', '').replace(' ', ''): col for col in data.columns}
        self.original_column_order = [col.replace('_', '').replace(' ', '') for col in data.columns]
        # store original data types
        self.original_dtypes = data.dtypes.to_dict()

        # attribute types
        self.cat_attrs = categorical_columns
        self.num_attrs = list(set(data.columns.tolist()) - set(self.cat_attrs))

        # convert categorical attributes to string
        data[self.cat_attrs] = data[self.cat_attrs].astype(str)

        # remove underscore in column names for correct inverse decoding
        data.columns = [col.replace('_', '').replace(' ', '') for col in data.columns]
        self.cat_attrs = [col.replace('_', '').replace(' ', '') for col in self.cat_attrs]
        self.num_attrs = [col.replace('_', '').replace(' ', '') for col in self.num_attrs]

        # iterate over categorical attributes
        for cat_attr in self.cat_attrs:
            # add col name to every categorical entry to make them distinguishable for embedding
            data[cat_attr] = cat_attr + '_' + data[cat_attr].astype('str')

        data = data[[*self.cat_attrs, *self.num_attrs]]

        # Initialize the quantile transformer if there are numerical attributes
        if self.num_attrs:
            self.num_scaler = QuantileTransformer(output_distribution='normal', random_state=self.seed)
            # fit transformation to numerical attributes
            self.num_scaler.fit(data[self.num_attrs])
            # transform numerical attributes
            train_num_scaled = self.num_scaler.transform(data[self.num_attrs])
            # convert numerical attributes
            train_num_torch = torch.FloatTensor(train_num_scaled)
        else:
            train_num_torch = None

        # get vocabulary of categorical attributes
        vocabulary_classes = np.unique(data[self.cat_attrs])

        # Initialize categorical attribute encoder 
        self.label_encoder = LabelEncoder()
        # fit encoder to categorical attributes
        self.label_encoder.fit(vocabulary_classes)
        # transform categorical attributes
        train_cat_scaled = data[self.cat_attrs].apply(self.label_encoder.transform)
        # convert categorical attributes
        train_cat_torch = torch.LongTensor(train_cat_scaled.values)

        # collect unique values of each categorical attribute
        self.vocab_per_attr = {cat_attr: set(train_cat_scaled[cat_attr]) for cat_attr in self.cat_attrs}

        # Create Tensor Dataset
        if train_num_torch is not None:
            train_set = TensorDataset(train_cat_torch, train_num_torch)
        else:
            train_set = TensorDataset(train_cat_torch)

        # Create DataLoader
        self.dataloader = DataLoader(
            dataset=train_set, # training dataset
            batch_size=batch_size, # training batch size
            num_workers=0, # number of workers
            shuffle=True # shuffle training data
        )

        # determine number unique categorical tokens
        n_cat_tokens = len(np.unique(data[self.cat_attrs]))
        # determine total categorical embedding dimension
        self.cat_dim = self.cat_emb_dim * len(self.cat_attrs)
        # determine total numerical embedding dimension
        num_dim = len(self.num_attrs) if self.num_attrs else 0
        # determine total embedding dimension
        self.encoded_dim = self.cat_dim + num_dim
        
        # initialize the FinDiff synthesizer model 
        self.synthesizer_model = MLPSynthesizer(
            d_in=self.encoded_dim,
            hidden_layers=self.mlp_layers,
            activation=activation,
            n_cat_tokens=n_cat_tokens,
            n_cat_emb=self.cat_emb_dim,
            embedding_learned=False
        )
        # initialize the FinDiff base diffuser model
        self.diffuser_model = BaseDiffuser(
            total_steps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
            scheduler=scheduler
        )
        # determine synthesizer model parameters
        parameters = filter(lambda p: p.requires_grad, self.synthesizer_model.parameters())
        # init Adam optimizer
        self.optimizer = optim.Adam(parameters, lr=learning_rate)
        # init learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, verbose=False)



    def fit(self):
        # int mean-squared-error loss
        loss_fnc = nn.MSELoss()

        # init collection of training epoch losses
        train_epoch_losses = []

        # set the model in training mode
        self.synthesizer_model.train()

        # init the training progress bar 
        pbar = tqdm(iterable=range(self.epochs), position=0, leave=True)

        # iterate over training epochs
        for epoch in pbar:

            base_params = {'epoch': epoch, 'seed': self.seed, 'mlp_layers': self.mlp_layers}

            # init epoch training batch losses
            batch_losses = []

            # iterate over epoch batches
            # for batch_cat, batch_num, batch_y in dataloader:       
            for batch in self.dataloader:
                if len(batch) == 2:
                    batch_cat, batch_num = batch
                else:
                    batch_cat = batch[0]
                    batch_num = None

                # sample diffusion timestep
                timesteps = self.diffuser_model.sample_random_timesteps(n=batch_cat.shape[0])

                # determine categorical embeddings
                batch_cat_emb = self.synthesizer_model.embed_categorical(x_cat=batch_cat)

                # concatenate categorical and numerical embeddings
                if batch_num is not None:
                    batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)
                else:
                    batch_cat_num = batch_cat_emb


                # add diffuser gaussian noise
                batch_noise_t, noise_t = self.diffuser_model.add_gauss_noise(x_num=batch_cat_num, t=timesteps)

                # conduct synthesizer model forward pass
                #predicted_noise = synthesizer_model(x=batch_noise_t, timesteps=timesteps)#, label=batch_y)
                predicted_noise = self.synthesizer_model(x=batch_noise_t, timesteps=timesteps)

                # compute training batch loss
                batch_loss = loss_fnc(input=noise_t, target=predicted_noise)

                # reset model gradients
                self.optimizer.zero_grad()

                # run model backward pass
                batch_loss.backward()

                # optimize model parameters
                self.optimizer.step()

                # collect training batch losses
                batch_losses.append(batch_loss.detach().cpu().numpy())

            # determine mean training epoch loss
            batch_losses_mean = np.mean(np.array(batch_losses))

            # update learning rate scheduler
            self.lr_scheduler.step()

            # collect mean training epoch loss
            train_epoch_losses.append(batch_losses_mean)

            # prepare and set training epoch progress bar update
            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            pbar.set_description('[LOG {}] epoch: {}, train-loss: {}'.format(str(now), str(epoch).zfill(4), str(batch_losses_mean)))

    def sample(self, num_samples: int):      
        # set number of diffusion steps
        diffusion_steps = 10
        samples = torch.randn((num_samples, self.encoded_dim))
                
        # init the generation progress bar 
        pbar = tqdm(iterable=reversed(range(0, diffusion_steps)), position=0, leave=True)

        # iterate over diffusion steps
        for diffusion_step in pbar:

            # prepare and set training epoch progress bar update
            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            pbar.set_description('[LOG {}] Diffusion Step: {}'.format(str(now), str(diffusion_step).zfill(4)))

            # init diffusion timesteps
            timesteps = torch.full((num_samples,), diffusion_step, dtype=torch.long)

            # run synthesizer model forward pass
            model_out = self.synthesizer_model(x=samples.float(), timesteps=timesteps) #, label=label_torch)

            # run diffuser model forward pass
            samples = self.diffuser_model.p_sample_gauss(model_out, samples, timesteps)

        # Decode generated data
        # split sample into numeric and categorical parts
        samples = samples.detach().numpy()
        if self.num_attrs:
            samples_num = samples[:, self.cat_dim:]
            samples_cat = samples[:, :self.cat_dim]
            # denormalize numeric attributes
            z_norm_upscaled = self.num_scaler.inverse_transform(samples_num)
            z_norm_df = pd.DataFrame(z_norm_upscaled, columns=self.num_attrs)
        else:
            samples_cat = samples
            z_norm_df = pd.DataFrame()

        # get embedding lookup matrix
        embedding_lookup = self.synthesizer_model.get_embeddings().cpu()

        # reshape back to batch_size * n_dim_cat * cat_emb_dim
        samples_cat = samples_cat.reshape(-1, len(self.cat_attrs), self.cat_emb_dim)

        # compute pairwise distances
        distances = torch.cdist(x1=embedding_lookup, x2=torch.Tensor(samples_cat))

        # get the closest distance based on the embeddings that belong to a column category
        z_cat_df = pd.DataFrame(index=range(len(samples_cat)), columns=self.cat_attrs)

        nearest_dist_df = pd.DataFrame(index=range(len(samples_cat)), columns=self.cat_attrs)

        # iterate over categorical attributes
        for attr_idx, attr_name in enumerate(self.cat_attrs):

            attr_emb_idx = list(self.vocab_per_attr[attr_name])
            attr_distances = distances[:, attr_emb_idx, attr_idx]

            nearest_values, nearest_idx = torch.min(attr_distances, dim=1)
            nearest_idx = nearest_idx.cpu().numpy()

            z_cat_df[attr_name] = np.array(attr_emb_idx)[nearest_idx]  # need to map emb indices back to column indices
            nearest_dist_df[attr_name] = nearest_values.cpu().numpy()

        z_cat_df = z_cat_df.apply(self.label_encoder.inverse_transform)

        # Remove prepended category names
        for col in z_cat_df.columns:
            z_cat_df[col] = z_cat_df[col].str.replace(f"{col}_", "")

        samples_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)
        
        # Reorder columns based on the original column order
        samples_decoded = samples_decoded[self.original_column_order]

        # Rename columns back to original names
        samples_decoded.columns = [self.original_column_names[col] for col in samples_decoded.columns]
        
         # Match the original data types
        for col in samples_decoded.columns:
            samples_decoded[col] = samples_decoded[col].astype(self.original_dtypes[col])

        return samples_decoded
        
                                