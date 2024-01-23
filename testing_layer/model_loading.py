from torchvision.models import vit_l_16, ViT_L_16_Weights, resnet152, ResNet152_Weights, \
efficientnet_v2_m, EfficientNet_V2_M_Weights, convnext_large, ConvNeXt_Large_Weights, vgg16, VGG16_Weights
from .workflows.enums import SupportedModels
# from torchsummary import summary
network_features = []


supported_models = {
    SupportedModels.VGG: vgg16,
    SupportedModels.CONVNEXT: convnext_large,
    SupportedModels.RESNET: resnet152,
    SupportedModels.VIT: vit_l_16,
    SupportedModels.EFFICIENTNET: efficientnet_v2_m
}
supported_weights = {
    SupportedModels.VGG: VGG16_Weights.IMAGENET1K_V1,
    SupportedModels.CONVNEXT: ConvNeXt_Large_Weights.IMAGENET1K_V1,
    SupportedModels.RESNET: ResNet152_Weights.IMAGENET1K_V2,
    SupportedModels.VIT: ViT_L_16_Weights.IMAGENET1K_V1,
    SupportedModels.EFFICIENTNET: EfficientNet_V2_M_Weights.IMAGENET1K_V1
}

def load_model(chosen_model: SupportedModels):
    """Load one of the researched models from pytorch repository."""
    weights = supported_weights[chosen_model]

    model = supported_models[chosen_model](weights=weights)
    # print(model)
    return model, weights.transforms()


def collect_features(model, input, output):
    global network_features
    network_features.append(input[0].detach().cpu().numpy())

#TODO : Przetestuj jakie są wielkości

def setup_a_hook(model, chosen_model: SupportedModels):
    if chosen_model == SupportedModels.CONVNEXT:
        return model.classifier[-1].register_forward_hook(collect_features)
    elif chosen_model == SupportedModels.EFFICIENTNET:
        return model.classifier[-1].register_forward_hook(collect_features)
    elif chosen_model == SupportedModels.VGG:
        return model.classifier[-1].register_forward_hook(collect_features)
    elif chosen_model == SupportedModels.RESNET:
        return model.fc.register_forward_hook(collect_features)
    elif chosen_model == SupportedModels.VIT:
        return model.heads.register_forward_hook(collect_features)


def remove_hook(hook):
    hook.remove()





