# Diffusion Prior components from MindEyeV2
# Reference: 
# https://github.com/MedARC-AI/MindEyeV2/blob/main/src/models.py
# FlaggedCausalTransformer
# PriorNetwork
# BrainDiffusionPrior (subclass of DiffusionPrior from dalle2_pytorch)

import random
import torch
import torch.nn as nn
from tqdm import tqdm

# Import from diffusers
# Please make sure to install diffusers library
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import (
    l2norm,
    default,
    exists,
    RotaryEmbedding,
    SinusoidalPosEmb,
    MLP,
    Rearrange,
    repeat,
    rearrange,
    prob_mask_like,
    LayerNorm,
    RelPosBias,
    Attention,
    FeedForward,
)

# FlaggedCausalTransformer Module
class FlaggedCausalTransformer(nn.Module):

    """
    Transformer with relative positional bias and optional rotary embeddings,
    with the option to return the output of each layer in addition to the final output.
    The transformer is causal by default, but can be made non-causal by passing 
    causal=False on initialization.

    NOTE: This module is used in PriorNetwork
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        norm_in: bool = False,
        norm_out: bool = True,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        final_proj: bool = True,
        normformer: bool = False,
        rotary_emb: bool = True,
        causal: bool = True
    ):

        """
        Args:
        - dim: dimension of the input and output of the transformer
        - depth: number of layers in the transformer
        - dim_head: dimension of each attention head
        - heads: number of attention heads
        - ff_mult: multiplier for the hidden dimension of the feedforward network
        - norm_in: whether to apply layer normalization to the input of the transformer
        - norm_out: whether to apply layer normalization to the output of the transformer
        - attn_dropout: dropout rate for the attention layers
        - ff_dropout: dropout rate for the feedforward layers
        - final_proj: whether to apply a final linear projection to the output of the transformer
        - normformer: whether to apply layer normalization after the activation function in the feedforward network
        - rotary_emb: whether to use rotary positional embeddings in the attention layers
        - causal: whether the transformer should be causal (i.e. whether to apply a causal mask in attention)
        """

        super().__init__()

        # from latest BLOOM model and Yandex's YaLM
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        # relative positional bias, which is important for generalization to different sequence lengths than seen at training time
        self.rel_pos_bias = RelPosBias(heads = heads)
        # rotary positional embeddings, which are also important for generalization to different sequence lengths
        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        # build transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        # final layer norm and projection
        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final de-noised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x is expected to be in shape [batch, sequence_length, dim]
        # also get device
        n, device = x.shape[1], x.device

        # initial layer norm
        x = self.init_norm(x)

        # get relative positional bias for the sequence length of x, which will be used in each attention layer
        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        # pass through each layer of the transformer
        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        # final layer norm and projection
        out = self.norm(x)
        return self.project_out(out)

# PriorNetwork Module
class PriorNetwork(nn.Module):

    """
    PriorNetwork: 
    The PriorNetwork takes in an image embedding, a diffusion timestep, 
    and optionally a brain embedding and text embedding, and predicts the 
    image embedding at the previous diffusion timestep. 
    The PriorNetwork is trained with a standard DDPM objective, 
    where the model is given a noisy image embedding at a random diffusion 
    timestep and is trained to predict the original image embedding. 
    The PriorNetwork uses a FlaggedCausalTransformer to attend over the 
    image embedding, brain embedding, time embedding, and optionally learned 
    query embeddings. The output of the transformer is then projected to 
    predict the image embedding at the previous diffusion timestep. 
    The PriorNetwork also includes classifier free guidance by randomly 
    dropping out the brain and image embeddings during training and 
    replacing them with learned null embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_timesteps: int = None,
        num_time_embeds: int = 1,
        # num_image_embeds: int = 1,
        # num_brain_embeds: int = 1,
        num_tokens: int = 257,
        causal: bool = True,
        learned_query_mode: str = 'none',
        **kwargs
    ):
        
        """
        Args:
        - dim: dimension of the image and brain embeddings, as well as the time embeddings and the input and output of the transformer
        - num_timesteps: number of diffusion timesteps, which determines the number of time embeddings; if None, will use continuous time embeddings with a 2 layer MLP instead of learned embeddings
        - num_time_embeds: number of time embeddings to use; 
                           the time embeddings will be repeated num_time_embeds 
                           times and concatenated to the image and brain embeddings 
                           before being passed to the transformer; more time embeddings 
                           allows for more capacity for the model to learn about the 
                           diffusion timestep, at the cost of more compute;
        - num_tokens: number of tokens in the image and brain embeddings;
        - causal: whether the transformer should be causal (i.e. whether to apply a causal mask in attention)
        """

        super().__init__()

        # set attributes
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode
        self.num_tokens = num_tokens
        self.self_cond = False

        # time embedding module, which can be either learned embeddings 
        # or continuous embeddings with a 2 layer MLP; 
        # the time embeddings will be repeated num_time_embeds times and 
        # concatenated to the image and brain embeddings before being 
        # passed to the transformer
        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # learned query embeddings, 
        # which are optionally added to the input of the transformer as 
        # additional tokens;
        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
        
        # the transformer that attends over the image embedding, 
        # brain embedding, time embedding, and optionally learned query 
        # embeddings;
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal = causal, **kwargs)

        # null embeddings for classifier free guidance, 
        # which are used to replace the brain and image embeddings with 
        # learned null embeddings with some probability during training;
        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale: float = 1.,
        **kwargs
    ) -> torch.Tensor:
        
        """
        Forward method with classifier free guidance scaling.

        NOTE:
        args and kwargs are passed to the forward method, 
        which should include the image embedding, 
        diffusion timesteps, and optionally the brain embedding 
        and text embedding, as well as the classifier free guidance 
        dropout probabilities for the brain and image embeddings.

        Args:
            - cond_scale: the scale for classifier free guidance;
        """

        logits = self.forward(*args, **kwargs)

        # if cond_scale is 1, just return the logits; 
        # otherwise, compute the null logits by passing in the 
        # same inputs but with brain_cond_drop_prob and 
        # image_cond_drop_prob set to 1 
        # (i.e. always drop out the brain and image embeddings), 
        # and then return the scaled combination of the logits 
        # and null logits
        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
        *,
        self_cond: torch.Tensor = None,
        brain_embed: torch.Tensor = None,
        text_embed: torch.Tensor = None,
        brain_cond_drop_prob: float = 0.,
        text_cond_drop_prob: float = None,
        image_cond_drop_prob: float = 0.
    ) -> torch.Tensor:
        
        # if text_embed is not None, use it for brain_embed, 
        # since the transformer will attend over it in the same way 
        # as the brain embedding;
        if text_embed is not None:
            brain_embed = text_embed
        # if text_cond_drop_prob is not None, use it for 
        # brain_cond_drop_prob
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob
        
        # image_embed = image_embed.view(len(image_embed),-1,16*16)
        # text_embed = text_embed.view(len(text_embed),-1,768)
        # brain_embed = brain_embed.view(len(brain_embed),-1,16*16)
        # print(*image_embed.shape)
        # print(*image_embed.shape, image_embed.device, image_embed.dtype)
        
        batch, _ , dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds
        
        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device = device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings
        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain 
        # encodings are available for attention (for classifier free guidance, 
        # even though it seems from the paper it was not used in the prior ddpm, 
        # as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds.forward(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b = batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device = brain_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            learned_queries = torch.empty((batch, 0, dim), device = brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device = brain_embed.device)
        
        tokens = torch.cat((
            brain_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim = -2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer.forward(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed

# BrainDiffusionPrior Module
class BrainDiffusionPrior(DiffusionPrior):
    
    """ 
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """

    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.,
                generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
            # noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

        # print("PS removed all image_embed_scale instances!")
        image_embed = normalized_image_embed #/ self.image_embed_scale
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)
        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale, 
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = nn.functional.mse_loss(pred, target) # mse
        # print("1", loss)
        # loss += (1 - nn.functional.cosine_similarity(pred, target).mean())
        # print("2", (1 - nn.functional.cosine_similarity(pred, target).mean()))
        return loss, pred

    def forward(
        self,
        text = None,
        image = None,
        voxel = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ) -> (tuple[torch.Tensor, torch.Tensor]):
        
        assert exists(text) ^ exists(text_embed) ^ exists(voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            if self.voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse = clip_voxels
            # text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # PS: I dont think we need this? also if uncommented this does in-place global variable change
        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)
        
        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred


# Alias for backward compatibility
BrainDiffusionPriorOld = BrainDiffusionPrior

# test code
if __name__ == "__main__":

    # setup diffusion prior network
    out_dim = 1664
    depth = 6
    dim_head = 52
    heads = out_dim // 52 # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
            dim = out_dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            causal = False,
            num_tokens = 256,
            learned_query_mode = "pos_emb"
        )

    diffusion_prior = BrainDiffusionPrior(
        net = prior_network,
        image_embed_dim  = out_dim,
        condition_on_text_encodings = False,
        timesteps = timesteps,
        cond_drop_prob = 0.2,
        image_embed_scale = None,
    )

    # Test forward pass with random data
    out = diffusion_prior.forward(
        text_embed = torch.randn(2, 256, out_dim),
        image_embed = torch.randn(2, 256, out_dim)
    )

    # print output shape
    print("Output loss:", out[0])  # loss
    print("Predicted image embedding shape:", out[1].shape)  # predicted image

    # Print outs
    # Output loss: tensor(1.3351, grad_fn=<MseLossBackward0>)
    # Predicted image embedding shape: torch.Size([2, 256, 1664])