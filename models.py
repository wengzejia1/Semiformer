import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg

from timm.models.registry import register_model

from vision_transformer import VisionTransformer, _cfg
from resnet_model import myResnet50, myResnet101, myResnet152
from semiformer_convmixer import Semiformer_convmixer
from semiformer import Semiformer
from semiformer_PE import Semiformer_PE


@register_model
def Semiformer_small_patch16(pretrained=False, **kwargs):
    model = Semiformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Semiformer_PE_small_patch16(pretrained=False, **kwargs):
    model = Semiformer_PE(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def Semiformer_convmixer_12depth(pretrained=False, **kwargs):
    model = Semiformer_convmixer(convmixer_dim=768, embed_dim=384, depth=12,
            num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)

    if pretrained:
        raise NotImplementedError

    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_med_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def myresnet50(pretrained=False, num_classes=1000, **kwargs):
    model = myResnet50(num_classes=num_classes)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def myresnet101(pretrained=False, num_classes=1000, **kwargs):
    model = myResnet101(num_classes=num_classes, )
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def myresnet152(pretrained=False, num_classes=1000, **kwargs):
    model = myResnet152(num_classes=num_classes, )
    if pretrained:
        raise NotImplementedError
    return model


