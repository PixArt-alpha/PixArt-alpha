import torchvision.transforms as T

TRANSFORMS = {}


def register_transform(transform):
    name = transform.__name__
    if name in TRANSFORMS:
        raise RuntimeError(f'Transform {name} has already registered.')
    TRANSFORMS.update({name: transform})


def get_transform(type, resolution):
    transform = TRANSFORMS[type](resolution)
    transform = T.Compose(transform)
    transform.image_size = resolution
    return transform


@register_transform
def default_train(n_px):
    return [
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(n_px),  # Image.BICUBIC
        T.CenterCrop(n_px),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
