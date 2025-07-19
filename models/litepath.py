import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        q = self.proj_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.proj_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim_dict):
        super(Projector, self).__init__()
        self.projectors = nn.ModuleDict({k: nn.Linear(in_dim, v) for k, v in out_dim_dict.items()})

    def forward(self, x):
        """
        Forward pass through the projector.
        :param x: Input features
        :return: Dictionary of projected outputs
        """
        outputs = {k: proj(x) for k, proj in self.projectors.items()}
        return outputs


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, proj_dim=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), out_dim_dict=None, extract_block=None):
        super(VisionTransformer, self).__init__()
        self.extract_block = extract_block
        self.proj_dim = proj_dim
        self.num_features = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if proj_dim == 0:
            self.head = nn.Identity()
        else:
            self.head = nn.Sequential(nn.Linear(embed_dim, proj_dim),
                                      nn.GELU(),
                                      norm_layer(proj_dim))

        feat_dim = embed_dim if proj_dim == 0 else proj_dim
        self.projector = Projector(feat_dim, out_dim_dict) if out_dim_dict else None

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.extract_block is not None and i == self.extract_block:
                cls_token = x[:, 0]  # bs, embed_dim
                patch_tokens = x[:, 1:]  # bs, num, embed_dim
                patch_token = patch_tokens.mean(1)  # bs, embed_dim
                embedding = torch.cat([cls_token, patch_token], dim=-1)  # bs, embed_dim * 2
                return embedding

        x = self.norm(x)
        x = self.head(x[:, 0])
        x = self.projector(x) if self.projector is not None else x
        if isinstance(x, dict):
            x = next(iter(x.values()))
        return x


def custom_vit_tiny_patch16_224(device, ckpt_path, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model


def custom_vit_small_patch16_224(device, ckpt_path, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = custom_vit_tiny_patch16_224(device, '/nfs/usrhome/xhuangbs/ycaibt/CPath/PrePATH/models/ckpts/litepath_wide_no_projector.pth', proj_dim=1024)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(output.shape)